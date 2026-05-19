import sys
from matplotlib import pyplot as plt
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Retrieval
from utils.tools import EarlyStopping

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from sklearn.manifold import TSNE

import os
import time
import warnings
warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model = Retrieval.Model(self.args).float()

        if self.args.use_multi_gpu:
            self.device = torch.device('cuda:{}'.format(self.args.rank))
            model = DDP(model.cuda(), device_ids=[self.args.rank],output_device=self.args.rank,find_unused_parameters=True)   
        else:
            self.device = self.args.gpu
            model = model.to(self.device)
        
        if not self.args.is_pretraining:
            # load pretrain model
            self.retrieval_model = Retrieval.Model(self.args,is_pretrain=True).float()
            self.retrieval_model.load_state_dict(torch.load(os.path.join('./checkpoints/' + self.args.retrieval_data+'/'+str(self.args.pred_len), 'checkpoint.pth')))

            for param in self.retrieval_model.parameters():
                param.requires_grad = False  # not update by gradient
            
            self.retrieval_model = self.retrieval_model.to(self.device)
            self.retrieval_model.eval()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def sequence_similarity(self,seq_1,seq_2):
        # normalization
        if self.args.normalization:
            means = seq_1.mean(1, keepdim=True).detach()
            seq_1 = seq_1 - means
            stdev = torch.sqrt(torch.var(seq_1, dim=1, keepdim=True, unbiased=False) + 1e-5)
            seq_1 /= stdev

        seq_1 = seq_1.permute(0,2,1)
        seq_2 = seq_2.permute(0,2,1)
        sim = torch.zeros((seq_1.shape[0],seq_2.shape[0],seq_1.shape[1])).to(self.device) #[bs,retrieval_size,n_var]
        for j in range(seq_1.shape[1]):
            a = seq_1[:,j,:]
            b = seq_2[:,j,:]
            sim[:,:,j] = torch.mul(a.unsqueeze(1),b.unsqueeze(0)).sum(dim=-1) 
        
        return sim

    def vali(self, vali_data, vali_loader, dataset_y, x_cls):
        total_loss = []
        total_count = []
        self.model.eval()
        with torch.no_grad():
            idx = torch.arange(1,dataset_y.shape[0]+1).to(self.device)
            idx = 1/torch.log2(idx.float()+1)
            idx = idx.unsqueeze(0)
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        cls_token = self.model(batch_x)
                else:
                    cls_token= self.model(batch_x)  
                
                sim_y =self.sequence_similarity(batch_y[:,-self.args.pred_len:,:],dataset_y) #[bs,retrieval_size,n_var]
                cos_sim = torch.zeros((cls_token.shape[0],x_cls.shape[0],cls_token.shape[1])).to(self.device) #[bs,retrieval_size,n_var]
                for j in range(cls_token.shape[1]):
                    a = cls_token[:,j,:]
                    b = x_cls[:,j,:]
                    cos_sim[:,:,j] = torch.mul(a.unsqueeze(1),b.unsqueeze(0)).sum(dim=-1) 
                
                cos_sim = cos_sim.permute(0,2,1)
                sim_y = sim_y.permute(0,2,1)
                sim_y = sim_y.reshape(-1,sim_y.shape[-1])
                cos_sim = cos_sim.reshape(-1,cos_sim.shape[-1])
                
                sim_sorted,sim_idx = torch.sort(cos_sim,dim=1,descending=True)
                sim_true,sim_idx_true = torch.sort(sim_y,dim=1,descending=True)
                # Normalized Discounted Cumulative Gain, NDCG
                IDCG = torch.sum(sim_true * idx,dim=1)
                sim_ = sim_y.gather(dim=1,index=sim_idx)
                DCG = torch.sum(sim_ * idx,dim=1)
                NDCG = DCG/IDCG
                
                NDCG = torch.nan_to_num(NDCG,nan=0)
                total_loss += NDCG.cpu().tolist()
        total_loss = torch.tensor(np.average(total_loss)).to(self.device)

        if self.args.use_multi_gpu:
            dist.barrier()   
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        retrieval_data, retrieval_loader = self._get_data(flag='retrieval')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)

        model_optim = self._select_optimizer()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            # encode retrieval data and load the prediction part
            dataset_y = [] # retrieve dataset of the prediction part
            x_cls = [] # encoded the retrieval data
            for i, (batch_x_r, batch_y_r) in enumerate(retrieval_loader):
                # retrieve dataset of the prediction part
                dataset_y += batch_y_r[:,-self.args.pred_len:,:].tolist() 
                
                # encoded the retrieval data
                batch_retrieval = torch.cat([batch_x_r, batch_y_r[:,-self.args.pred_len:,:]], dim=1)
                batch_retrieval = batch_retrieval.float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        cls_token = self.retrieval_model(batch_retrieval)
                else:
                    cls_token = self.retrieval_model(batch_retrieval)
                cls_token = cls_token.detach().cpu()
                x_cls.append(cls_token)

            dataset_y = torch.tensor(dataset_y).to(self.device) # [retrieval_size,pred_len,n_var]
            x_cls = torch.cat(x_cls, dim=0).to(self.device).detach() # [retrieval_size,n_var,d_model]
            
            if self.args.normalization:
                means = dataset_y.mean(1, keepdim=True).detach()
                dataset_y = dataset_y - means
                stdev = torch.sqrt(torch.var(dataset_y, dim=1, keepdim=True, unbiased=False) + 1e-5)
                dataset_y /= stdev

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # similarity of the predictin part
                sim = self.sequence_similarity(batch_y[:,-self.args.pred_len:,:],dataset_y) #[bs,retrieval_size,n_var]
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        cls_token = self.model(batch_x)
                else:
                    cls_token= self.model(batch_x)  

                # cosine similarity of the encoded representations
                cos_sim = torch.zeros((cls_token.shape[0],x_cls.shape[0],cls_token.shape[1])).to(self.device) #[bs,retrieval_size,n_var]
                for j in range(cls_token.shape[1]):
                    a = cls_token[:,j,:]
                    b = x_cls[:,j,:]
                    cos_sim[:,:,j] = torch.mul(a.unsqueeze(1),b.unsqueeze(0)).sum(dim=-1) 
                
                #listmle loss
                sim_sorted,sim_idx = torch.sort(sim,dim=1,descending=True)
                pred_sorted_by_true = cos_sim.gather(dim=1, index=sim_idx)
                log_cumsum_exp = torch.logcumsumexp(pred_sorted_by_true.flip([1]), dim=1).flip([1])
                loss = torch.sum(log_cumsum_exp-pred_sorted_by_true, dim=1).mean()
               
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.rank == 0) or not self.args.use_multi_gpu:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                
            if (self.args.use_multi_gpu and self.args.rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = torch.tensor(np.average(train_loss)).to(self.device)

            if self.args.use_multi_gpu:
                dist.barrier()   
                dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
                train_loss = train_loss.item() / dist.get_world_size()

            vali_loss = self.vali(vali_data, vali_loader, dataset_y, x_cls)
            test_loss = self.vali(test_data, test_loader, dataset_y, x_cls)
            if (self.args.use_multi_gpu and self.args.rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if (self.args.use_multi_gpu and self.args.rank == 0) or not self.args.use_multi_gpu:
                    print("Early stopping")
                    break
            if self.args.use_multi_gpu:
                train_loader.sampler.set_epoch(epoch + 1)
                retrieval_loader.sampler.set_epoch(epoch + 1)

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            self.model.module.load_state_dict(torch.load(best_model_path))
        else:
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def vali_pretrain(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = torch.cat([batch_x, batch_y[:,-self.args.pred_len:,:]], dim=1)
                batch_x = batch_x.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        y_hat,y = self.model(batch_x,use_mask=True)
                else:
                    y_hat,y = self.model(batch_x,use_mask=True)

                loss = criterion(y_hat, y)

                total_loss.append(loss.cpu().item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train_pretrain(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        print('number of model params', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        path = os.path.join(self.args.checkpoints, self.args.data+'/'+str(self.args.pred_len))
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader): # batch_x: (batch_size, seq_len, n_var)
                iter_count += 1
                model_optim.zero_grad()

                batch_x = torch.cat([batch_x, batch_y[:,-self.args.pred_len:,:]], dim=1)
                batch_x = batch_x.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        y_hat,y = self.model(batch_x,use_mask=True)
                else:
                    y_hat,y = self.model(batch_x,use_mask=True)
                
                loss = criterion(y_hat, y)
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                train_loss.append(loss.cpu().item())
        
            train_loss = np.average(train_loss)
            vali_loss = self.vali_pretrain(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def test(self,setting, test=0):  
        test_data, test_loader = self._get_data(flag='test')
        retrieval_data, retrieval_loader = self._get_data(flag='retrieval')

        if self.args.use_multi_gpu:
            if self.args.rank == 0:
                print('loading model')
            self.model.module.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')),strict=False)
        else:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')),strict=False)

        self.model.eval()
        with torch.no_grad():
            dataset_y = [] # retrieve dataset of the prediction part
            x_cls = [] # encoded the retrieval data
            total_NDCG = []
            total_correlation = []
            for i, (batch_x_r, batch_y_r) in enumerate(retrieval_loader):
                # retrieve dataset of the prediction part
                dataset_y += batch_y_r[:,-self.args.pred_len:,:].tolist() 
                
                # encoded the retrieval data
                batch_retrieval = torch.cat([batch_x_r, batch_y_r[:,-self.args.pred_len:,:]], dim=1)
                batch_retrieval = batch_retrieval.float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        cls_token = self.retrieval_model(batch_retrieval)
                else:
                    cls_token = self.retrieval_model(batch_retrieval)
                cls_token = cls_token.detach().cpu()
                x_cls.append(cls_token)

            dataset_y = torch.tensor(dataset_y).to(self.device) # [retrieval_size,pred_len,n_var]
            x_cls = torch.cat(x_cls, dim=0).to(self.device).detach() # [retrieval_size,nvars,d_model]

            if self.args.normalization:
                means = dataset_y.mean(1, keepdim=True).detach()
                dataset_y = dataset_y - means
                stdev = torch.sqrt(torch.var(dataset_y, dim=1, keepdim=True, unbiased=False) + 1e-5)
                dataset_y /= stdev

            idx = torch.arange(1,dataset_y.shape[0]+1).to(self.device)
            idx = 1/torch.log2(idx.float()+1)
            idx = idx.unsqueeze(0)

            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        cls_token = self.model(batch_x)
                else:
                    cls_token= self.model(batch_x)  

                # predicted similarity
                cos_sim = torch.zeros((cls_token.shape[0],x_cls.shape[0],cls_token.shape[1])).to(self.device) #[bs,retrieval_size,n_var]
                for j in range(cls_token.shape[1]):
                    a = cls_token[:,j,:]
                    b = x_cls[:,j,:]
                    cos_sim[:,:,j] = torch.mul(a.unsqueeze(1),b.unsqueeze(0)).sum(dim=-1) 

                sim_y = self.sequence_similarity(batch_y[:,-self.args.pred_len:,:],dataset_y)
                
                cos_sim = cos_sim.permute(0,2,1)
                sim_y = sim_y.permute(0,2,1)
                sim_y = sim_y.reshape(-1,sim_y.shape[-1])
                cos_sim = cos_sim.reshape(-1,cos_sim.shape[-1])

                sim_sorted,sim_idx = torch.sort(cos_sim,dim=1,descending=True)
                sim_true,sim_idx_true = torch.sort(sim_y,dim=1,descending=True)
                # Normalized Discounted Cumulative Gain, NDCG
                IDCG = torch.sum(sim_true * idx,dim=1)
                sim_ = sim_y.gather(dim=1,index=sim_idx)
                DCG = torch.sum(sim_ * idx,dim=1)
                NDCG = DCG/IDCG
                NDCG = torch.nan_to_num(NDCG,nan=0)
                total_NDCG += NDCG.cpu().tolist()
                
        total_NDCG = torch.tensor(np.average(total_NDCG)).to(self.device)
        if self.args.use_multi_gpu:
            dist.barrier()   
            dist.all_reduce(total_NDCG, op=dist.ReduceOp.SUM)
            total_NDCG= total_NDCG.item() / dist.get_world_size()
        
        if (self.args.use_multi_gpu and self.args.rank == 0) or not self.args.use_multi_gpu:
            print('NDCG: {}'.format(total_NDCG))
        return


    def plot(self,setting):
        test_data, test_loader = self._get_data(flag='test')
        retrieval_data, retrieval_loader = self._get_data(flag='retrieval')

        if self.args.use_multi_gpu:
            if self.args.rank == 0:
                print('loading model')
            self.model.module.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')),strict=False)
        else:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')),strict=False)

        self.model.eval()
        with torch.no_grad():
            x_cls = [] # encoded the retrieval data
            dataset_x = []
            dataset_y = []
            dataset = []
            for i, (batch_x_r, batch_y_r) in enumerate(retrieval_loader):
                # retrieve dataset of the prediction part
                dataset_x += batch_x_r.tolist()
                dataset_y += batch_y_r[:,-self.args.pred_len:,:].tolist()
                dataset+=torch.cat((batch_x_r,batch_y_r[:,-self.args.pred_len:,:]),dim=1).cpu().tolist()
                # encoded the retrieval data
                batch_retrieval = torch.cat([batch_x_r, batch_y_r[:,-self.args.pred_len:,:]], dim=1)
                batch_retrieval = batch_retrieval.float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        cls_token = self.retrieval_model(batch_retrieval)
                else:
                    cls_token = self.retrieval_model(batch_retrieval)
                cls_token = cls_token.detach().cpu()
                x_cls.append(cls_token)

            dataset_x = torch.tensor(dataset_x).to(self.device)
            dataset_y = torch.tensor(dataset_y).to(self.device)
            dataset = torch.tensor(dataset).to(self.device)
            x_cls = torch.cat(x_cls, dim=0).to(self.device).detach() # [retrieval_size,nvars,d_model]

            if self.args.normalization:
                means = dataset_x.mean(1, keepdim=True).detach()
                dataset_x = dataset_x - means
                stdev = torch.sqrt(torch.var(dataset_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
                dataset_x /= stdev

            var = 0
            bs = 50
            for i, (batch_x, batch_y) in enumerate(test_loader):
                if i < 5:
                    continue
                batch_x = batch_x.float().to(self.device) # bs,seq_len,n_var
                batch_y = batch_y.float().to(self.device)

                for b in range(self.args.batch_size):
                    x = batch_x[b,:,var].cpu().tolist()
                    y = batch_y[b,-self.args.pred_len-1:,var].cpu().tolist()

                    axes_history = torch.arange(0,self.args.seq_len).tolist()
                    axes_prediction = torch.arange(self.args.seq_len-1,self.args.seq_len+self.args.pred_len).tolist()
                    
                    # 绘制x和y分别两种颜色的图
                    plt.figure()
                    plt.plot(axes_history, x, label="historical parts", color="steelblue",linewidth=2)
                    plt.plot(axes_prediction, y, label="prediction parts", color="lightskyblue",linewidth=2)
                    plt.rc('legend', fontsize=15)
                    plt.legend(loc='lower right')
                    plt.savefig('./input_data_'+str(b)+'.pdf', bbox_inches='tight')
                
                x = batch_x[bs,:,var].cpu().tolist()
                y = batch_y[bs,-self.args.pred_len-1:,var].cpu().tolist()

                axes_history = torch.arange(0,self.args.seq_len).tolist()
                axes_prediction = torch.arange(self.args.seq_len-1,self.args.seq_len+self.args.pred_len).tolist()
                
                # 绘制x和y分别两种颜色的图
                plt.figure()
                plt.plot(axes_history, x, label="historical parts", color="steelblue",linewidth=2)
                plt.plot(axes_prediction, y, label="prediction parts", color="lightskyblue",linewidth=2)
                plt.rc('legend', fontsize=15)
                plt.legend(loc='lower right')
                plt.savefig('./input_data.pdf', bbox_inches='tight')

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        cls_token = self.model(batch_x)
                else:
                    cls_token= self.model(batch_x)  

                cos_sim = torch.zeros((cls_token.shape[0],x_cls.shape[0],cls_token.shape[1])).to(self.device) #[bs,retrieval_size,n_var]
                for j in range(cls_token.shape[1]):
                    a = cls_token[:,j,:]
                    b = x_cls[:,j,:]
                    cos_sim[:,:,j] = torch.mul(a.unsqueeze(1),b.unsqueeze(0)).sum(dim=-1) 

                sim_x = self.sequence_similarity(batch_x,dataset_x)
                sim_y = self.sequence_similarity(batch_y[:,-self.args.pred_len:,:],dataset_y)
                sim_sorted,sim_idx = torch.sort(cos_sim,dim=1,descending=True)
                sim_x,sim_idx_x = torch.sort(sim_x,dim=1,descending=True)

                carts_idx = sim_idx[bs,0,var]
                x_idx = sim_idx_x[bs,0,var]

                data = dataset[x_idx,:,var].cpu().tolist()
                data_x = data[:self.args.seq_len]
                data_y = data[self.args.seq_len-1:]
                plt.figure()
                plt.plot(axes_history, data_x, label="historical parts", color="steelblue",linewidth=2)
                plt.plot(axes_prediction, data_y, label="prediction parts", color="lightskyblue",linewidth=2)
                plt.rc('legend', fontsize=15)
                plt.legend(loc='lower right')
                plt.savefig('./x_cause.pdf', bbox_inches='tight')
                print("x_cause:",sim_y[bs,x_idx,var].cpu().item())
                data_carts = dataset[carts_idx,:,var].cpu().tolist()
                data_x = data_carts[:self.args.seq_len]
                data_y = data_carts[self.args.seq_len-1:]
                plt.figure()
                plt.plot(axes_history, data_x, label="historical parts", color="steelblue",linewidth=2)
                plt.plot(axes_prediction, data_y, label="prediction parts", color="lightskyblue",linewidth=2)
                plt.rc('legend', fontsize=15)
                plt.legend(loc='lower right')
                plt.savefig('./carts.pdf', bbox_inches='tight')
                print("carts:",sim_y[bs,carts_idx,var].cpu().item())
                sys.exit()

    def plot_sim(self,setting):
        test_data, test_loader = self._get_data(flag='train')
        retrieval_data, retrieval_loader = self._get_data(flag='retrieval')
        if self.args.use_multi_gpu:
            if self.args.rank == 0:
                print('loading model')
            self.model.module.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')),strict=False)
        else:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')),strict=False)

        self.model.eval()
        with torch.no_grad():
            dataset_y = [] # retrieve dataset of the prediction part
            x_cls = [] # encoded the retrieval data
            cls_list = []
            for i, (batch_x_r, batch_y_r) in enumerate(retrieval_loader):
                # retrieve dataset of the prediction part
                dataset_y += batch_y_r[:,-self.args.pred_len:,:].tolist() 
                
                # encoded the retrieval data
                batch_retrieval = torch.cat([batch_x_r, batch_y_r[:,-self.args.pred_len:,:]], dim=1)
                batch_retrieval = batch_retrieval.float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        cls_token = self.retrieval_model(batch_retrieval)
                else:
                    cls_token = self.retrieval_model(batch_retrieval)
                cls_token = cls_token.detach().cpu()
                x_cls.append(cls_token)
                cls_list.append(cls_token)

            dataset_y = torch.tensor(dataset_y).to(self.device) # [retrieval_size,pred_len,n_var]
            x_cls = torch.cat(x_cls, dim=0).to(self.device).detach() # [retrieval_size,nvars,d_model]

            if self.args.normalization:
                means = dataset_y.mean(1, keepdim=True).detach()
                dataset_y = dataset_y - means
                stdev = torch.sqrt(torch.var(dataset_y, dim=1, keepdim=True, unbiased=False) + 1e-5)
                dataset_y /= stdev

            idx = torch.arange(1,dataset_y.shape[0]+1).to(self.device)
            idx = 1/torch.log2(idx.float()+1)
            idx = idx.unsqueeze(0)

            sim_list = []
            sim_hat_list = []

            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        cls_token = self.model(batch_x)
                else:
                    cls_token= self.model(batch_x)  

                cls_list.append(cls_token.cpu())
                # predicted similarity
                cos_sim = torch.zeros((cls_token.shape[0],x_cls.shape[0],cls_token.shape[1])).to(self.device) #[bs,retrieval_size,n_var]
                for j in range(cls_token.shape[1]):
                    a = cls_token[:,j,:]
                    b = x_cls[:,j,:]
                    cos_sim[:,:,j] = torch.mul(a.unsqueeze(1),b.unsqueeze(0)).sum(dim=-1) 

                sim_y = self.sequence_similarity(batch_y[:,-self.args.pred_len:,:],dataset_y)
                sim_list.append(sim_y.cpu())
                sim_hat_list.append(cos_sim.cpu())

            sim_list = torch.cat(sim_list,dim=0)
            sim_hat_list = torch.cat(sim_hat_list,dim=0)
            cls_list = torch.cat(cls_list,dim=0)

            sim_list = sim_list[:,:,0]
            sim_hat_list = sim_hat_list[:,:,0]
            cls_list = cls_list[:,0,:]

            sim_list = sim_list.cpu().numpy()
            sim_hat_list = sim_hat_list.cpu().numpy()
            cls_list = cls_list.cpu().numpy()

            print(sim_list.shape,sim_hat_list.shape,cls_list.shape)

            # plot similarity
            # sim_list = sim_list.reshape(-1)
            # sim_hat_list = sim_hat_list.reshape(-1)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.scatter(sim_list[0],sim_list[1], c='r')
            plt.scatter(sim_hat_list[0],sim_list[1], c='b')
            plt.savefig('sim.png')

            # plot tsne
            tsne = TSNE(n_components=2, init='pca', random_state=0)
            result = tsne.fit_transform(cls_list)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.scatter(result[:len(retrieval_data), 0], result[:len(retrieval_data), 1], c='r')
            plt.scatter(result[len(retrieval_data):, 0], result[len(retrieval_data):, 1], c='b')
            plt.savefig('tsne.png')
        return