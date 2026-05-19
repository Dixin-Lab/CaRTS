import sys
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Prediction, Retrieval, GPT4TS
from utils.tools import EarlyStopping, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn.functional as F
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Prediction': Prediction,
            'GPT4TS': GPT4TS,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu:
            self.device = torch.device('cuda:{}'.format(self.args.rank))
            model = DDP(model.cuda(), device_ids=[self.args.rank],output_device=self.args.rank)   
        else:
            self.device = self.args.gpu
            model = model.to(self.device)

        #retrieval model
        self.model_retreival = Retrieval.Model(self.args, is_pretrain=True).float() # rerieval data encoder
        self.model_query = Retrieval.Model(self.args).float() #query data encode

        self.model_retreival.load_state_dict(torch.load(os.path.join(self.args.retrieval_model_path, 'checkpoint.pth')),strict=False)
        self.model_query.load_state_dict(torch.load(os.path.join(self.args.query_model_path, 'checkpoint.pth')),strict=False)
        
        #freeze the retrieval model
        for i, (name, param) in enumerate(self.model_retreival.named_parameters()):
            param.requires_grad = False
        for i, (name, param) in enumerate(self.model_query.named_parameters()):
            param.requires_grad = False
        
        self.model_retreival = self.model_retreival.to(self.device)
        self.model_query = self.model_query.to(self.device)
        
        self.model_query.eval()
        self.model_retreival.eval()

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

    def vali(self, vali_data, vali_loader, criterion, dataset, x_cls):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_y = batch_y.float().to(self.device)
                
                # encode the query data 
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        cls_token = self.model_query(batch_x).detach()
                else:
                    cls_token = self.model_query(batch_x).detach()

                # calculate the similarity between the query data and the retrieval data
                sim = torch.zeros((cls_token.shape[0],x_cls.shape[0],cls_token.shape[1])).to(self.device) #[bs,retrieval_size,n_var]
                for j in range(cls_token.shape[1]):
                    a = cls_token[:,j,:]
                    b = x_cls[:,j,:]
                    sim[:,:,j] = torch.mul(a.unsqueeze(1),b.unsqueeze(0)).sum(dim=-1) 
                sim_sorted,sim_idx = torch.sort(sim,dim=1,descending=True)

                # select the top k similar retrieval data as prompt
                prompt = []
                for num in range(self.args.topk):
                    idx = sim_idx[:,num,:]
                    prompt_data = []
                    for j in range(idx.shape[0]):
                        data = []
                        for k in range(idx.shape[1]):
                            data.append(dataset[idx[j,k],:,k])
                        data = torch.stack(data,dim=1)
                        prompt_data.append(data)
                    prompt_data = torch.stack(prompt_data,dim=0) 
                    prompt.append(prompt_data)      
                
                if len(prompt) != 0:
                    prompt = torch.stack(prompt,dim=0).to(self.device)
                else:
                    prompt =None
                # do prediction
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(prompt, batch_x)  
                else:
                    outputs = self.model(prompt, batch_x)
                    
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        
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
            os.makedirs(path,exist_ok=True)

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

            dataset = []
            x_cls = []

            # load and encode the retrieval data
            for i, (batch_x_r, batch_y_r) in enumerate(retrieval_loader):
                dataset+=torch.cat((batch_x_r,batch_y_r[:,-self.args.pred_len:,:]),dim=1).cpu().tolist()
                batch_x_r = torch.cat((batch_x_r,batch_y_r[:,-self.args.pred_len:,:]),dim=1)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        cls_token = self.model_retreival(batch_x_r.float().to(self.device)).cpu().detach()
                else:
                    cls_token = self.model_retreival(batch_x_r.float().to(self.device)).cpu().detach()
                x_cls.append(cls_token)

            dataset = torch.tensor(dataset).to(self.device) # [retrieval_size,seq_len+pred_len,n_var]
            x_cls = torch.cat(x_cls, dim=0).to(self.device).detach() # [retrieval_size,n_var,d_model]

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device) #bs,seq_len,n_vars
    
                batch_y = batch_y.float().to(self.device)

                # encode the query data 
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        cls_token = self.model_query(batch_x).detach()
                else:
                    cls_token = self.model_query(batch_x).detach()
                
                # calculate the similarity between the query data and the retrieval data
                sim = torch.zeros((cls_token.shape[0],x_cls.shape[0],cls_token.shape[1])).to(self.device) #[bs,retrieval_size,n_var]
                for j in range(cls_token.shape[1]):
                    a = cls_token[:,j,:]
                    b = x_cls[:,j,:]
                    sim[:,:,j] = torch.mm(a,b.t())
                
                sim_sorted,sim_idx = torch.sort(sim,dim=1,descending=True)
                
                # select the top k similar retrieval data as prompt
                prompt = []
                for num in range(self.args.topk):
                    idx = sim_idx[:,num,:]
                    prompt_data = []
                    for j in range(idx.shape[0]):
                        data = []
                        for k in range(idx.shape[1]):
                            data.append(dataset[idx[j,k],:,k])
                        data = torch.stack(data,dim=1)
                        prompt_data.append(data)
                    prompt_data = torch.stack(prompt_data,dim=0) 
                    prompt.append(prompt_data)
                if len(prompt) != 0:     
                    prompt = torch.stack(prompt,dim=0).to(self.device)
                else:
                    prompt =None

                # do prediction
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(prompt, batch_x)
                else:
                    outputs = self.model(prompt, batch_x)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                loss = criterion(outputs, batch_y)
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
                
            vali_loss = self.vali(vali_data, vali_loader, criterion, dataset, x_cls)
            test_loss = self.vali(test_data, test_loader, criterion, dataset, x_cls)

            if (self.args.use_multi_gpu and self.args.rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if (self.args.use_multi_gpu and self.args.rank == 0) or not self.args.use_multi_gpu:
                    print("Early stopping")
                break
        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            self.model.module.load_state_dict(torch.load(best_model_path))
        else:
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def batch_directional_accuracy(self, y_true, y_pred, eps=1e-8):
        """
        Batch Directional Accuracy (DA)

        Parameters
        ----------
        y_true : ndarray, shape (B, N)
            Ground truth time series
        y_pred : ndarray, shape (B, N)
            Predicted time series

        Returns
        -------
        da : float
            Mean directional accuracy over batch
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        assert y_true.shape == y_pred.shape
        assert y_true.ndim == 2

        # True and predicted changes
        delta_true = y_true[:, 1:] - y_true[:, :-1]
        delta_pred = y_pred[:, 1:] - y_true[:, :-1]

        correct = (delta_true * delta_pred) > 0

        # DA per sequence
        da_per_batch = correct.mean(axis=1)

        return da_per_batch.mean()

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        retrieval_data, retrieval_loader = self._get_data(flag='retrieval')
        retreival_dataset_size = len(retrieval_data)
        if test:
            if (self.args.use_multi_gpu and self.args.rank == 0) or not self.args.use_multi_gpu:
                print('loading model')
            if self.args.use_multi_gpu:
                self.model.module.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            else:
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path,exist_ok=True)

        # compute inference time
        begin = time.time()

        self.model.eval()
        with torch.no_grad():
            dataset = []
            x_cls = []
            time_list = []
            
            # load and encode the retrieval data
            for i, (batch_x_r, batch_y_r) in enumerate(retrieval_loader):
                dataset+=torch.cat((batch_x_r,batch_y_r[:,-self.args.pred_len:,:]),dim=1).cpu().tolist()
                batch_x_r = torch.cat((batch_x_r,batch_y_r[:,-self.args.pred_len:,:]),dim=1)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        cls_token = self.model_retreival(batch_x_r.float().to(self.device)).cpu().detach()
                else:
                    cls_token = self.model_retreival(batch_x_r.float().to(self.device)).cpu().detach()
                x_cls.append(cls_token)
                if len(dataset) >= retreival_dataset_size*self.args.portion:
                    break

            dataset = torch.tensor(dataset).to(self.device) # [retrieval_size,seq_len+pred_len,n_var]
            x_cls = torch.cat(x_cls, dim=0).to(self.device).detach() # [retrieval_size,n_var,d_model]
            print(dataset.shape)
            for i, (batch_x, batch_y) in enumerate(test_loader):
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # encode the query data 
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        cls_token = self.model_query(batch_x).detach()
                else:
                    cls_token = self.model_query(batch_x).detach()
                
                sim = torch.mul(cls_token.unsqueeze(1),x_cls.unsqueeze(0)).sum(dim=-1)
                sim_sorted,sim_idx = torch.sort(sim,dim=1,descending=True)
                
                prompt = []
                for num in range(self.args.topk):
                    idx = sim_idx[:,num,:]
                    prompt_data = []
                    for j in range(idx.shape[0]):
                        data = []
                        for k in range(idx.shape[1]):
                            data.append(dataset[idx[j,k],:,k])
                        data = torch.stack(data,dim=1)
                        prompt_data.append(data)
                    prompt_data = torch.stack(prompt_data,dim=0) 
                    prompt.append(prompt_data)     
                      
                if len(prompt) != 0:
                    prompt = torch.stack(prompt,dim=0).to(self.device)
                else:
                    prompt =None
                
                # select the top k similar retrieval data as prompt
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(prompt, batch_x)
                else:
                    outputs = self.model(prompt, batch_x)
                
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if self.args.visual and i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        

        end = time.time()
        time_list.append(end-begin)
        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)
        inputx = np.concatenate(inputx,axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        time_avg = np.average(time_list)
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path,exist_ok=True)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        
        preds_trans = preds.transpose(0, 2, 1)
        trues_trans = trues.transpose(0, 2, 1)
        preds_trans = preds_trans.reshape(-1,preds_trans.shape[-1])
        trues_trans = trues_trans.reshape(-1,trues_trans.shape[-1])
        print(preds.shape, trues.shape,preds_trans.shape,trues_trans.shape)

        da = self.batch_directional_accuracy(trues_trans, preds_trans)

        mae = torch.tensor(mae).to(self.device)
        mse = torch.tensor(mse).to(self.device)
        rse = torch.tensor(rse).to(self.device)
        time_avg = torch.tensor(time_avg).to(self.device)
        if self.args.use_multi_gpu:
            dist.barrier()   
            dist.all_reduce(mae, op=dist.ReduceOp.SUM)
            dist.all_reduce(mse, op=dist.ReduceOp.SUM)
            dist.all_reduce(rse, op=dist.ReduceOp.SUM)
            dist.all_reduce(time_avg, op=dist.ReduceOp.SUM)
            mae = mae.item() / dist.get_world_size()
            mse = mse.item() / dist.get_world_size()
            rse = rse.item() / dist.get_world_size()
            time_avg = time_avg.item() / dist.get_world_size()
        
        if (self.args.use_multi_gpu and self.args.rank == 0) or not self.args.use_multi_gpu:
            print('time_avg:{}'.format(time_avg))
            print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            print('da:{}'.format(da))
            f = open("result.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rse:{}, da:{}'.format(mse, mae, rse, da))
            f.write('\n')
            f.write('\n')
            f.close()
        return
    
    def trasfer_learning_test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        retrieval_data, retrieval_loader = self._get_data(flag='retrieval')
        if 'weather' in self.args.retrieval_data_path:
            flag = 1
        else:
            flag = 0
        self.args.retrieval_data = self.args.data
        self.args.retrieval_root_path = self.args.root_path
        self.args.retrieval_data_path = self.args.data_path
        retrieval_data_source, retrieval_loader_source = self._get_data(flag='retrieval')
        total_retrieval_data_count = len(retrieval_data_source)
        
        if (self.args.use_multi_gpu and self.args.rank == 0) or not self.args.use_multi_gpu:
            print('loading model')
        if self.args.use_multi_gpu:
            self.model.module.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        else:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path,exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            dataset = []
            x_cls = []
            cnt = 0
            # load and encode the retrieval data
            for i, (batch_x_r, batch_y_r) in enumerate(retrieval_loader):
                if flag == 0:
                    dataset+=torch.cat((batch_x_r,batch_y_r[:,-self.args.pred_len:,:]),dim=1).cpu().tolist()
                    batch_x_r = torch.cat((batch_x_r,batch_y_r[:,-self.args.pred_len:,:]),dim=1)
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            cls_token = self.model_retreival(batch_x_r.float().to(self.device)).cpu().detach()
                    else:
                        cls_token = self.model_retreival(batch_x_r.float().to(self.device)).cpu().detach()
                    x_cls+=cls_token.tolist()
                else:
                    data=torch.cat((batch_x_r,batch_y_r[:,-self.args.pred_len:,:]),dim=1).permute(0,2,1).reshape(-1,self.args.seq_len+self.args.pred_len).unsqueeze(-1)
                    data = data.repeat(1,1,self.args.enc_in)
                    dataset += data.cpu().tolist()
                    batch_x_r = torch.cat((batch_x_r,batch_y_r[:,-self.args.pred_len:,:]),dim=1)
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            cls_token = self.model_retreival(batch_x_r.float().to(self.device)).cpu().detach()
                    else:
                        cls_token = self.model_retreival(batch_x_r.float().to(self.device)).cpu().detach()
                    cls_token = cls_token.reshape(-1,self.args.d_model).unsqueeze(1)
                    cls_token = cls_token.repeat(1,self.args.enc_in,1)
                    x_cls+=cls_token.tolist()
                
                if len(dataset) >= total_retrieval_data_count*self.args.portion:
                    break
            dataset = dataset[:int(total_retrieval_data_count*self.args.portion)]
            x_cls = x_cls[:int(total_retrieval_data_count*self.args.portion)]
            

            model_retrieval_source = Retrieval.Model(self.args, is_pretrain=True).float() # rerieval data encoder
            model_retrieval_source.load_state_dict(torch.load(os.path.join("../Retrieval/checkpoints/"+self.args.data+"/"+str(self.args.pred_len),'checkpoint.pth')),strict=False)
            model_retrieval_source = model_retrieval_source.to(self.device)
            for i, (batch_x_r, batch_y_r) in enumerate(retrieval_loader_source):
                dataset+=torch.cat((batch_x_r,batch_y_r[:,-self.args.pred_len:,:]),dim=1).cpu().tolist()
                batch_x_r = torch.cat((batch_x_r,batch_y_r[:,-self.args.pred_len:,:]),dim=1)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        cls_token = model_retrieval_source(batch_x_r.float().to(self.device)).cpu().detach()
                else:
                    cls_token = model_retrieval_source(batch_x_r.float().to(self.device)).cpu().detach()
                x_cls+=cls_token.tolist()
                
                if len(dataset) >= total_retrieval_data_count:
                    break
            
            dataset = dataset[:total_retrieval_data_count]
            x_cls = x_cls[:total_retrieval_data_count]

            dataset = torch.tensor(dataset).to(self.device) # [retrieval_size,seq_len+pred_len,n_var]
            x_cls = torch.tensor(x_cls).to(self.device).detach()

            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encode the query data 
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        cls_token = self.model_query(batch_x).detach()
                else:
                    cls_token = self.model_query(batch_x).detach()

                # calculate the similarity between the query data and the retrieval data
                sim = torch.zeros((cls_token.shape[0],x_cls.shape[0],cls_token.shape[1])).to(self.device)
                for j in range(cls_token.shape[1]):
                    a = cls_token[:,j,:]
                    b = x_cls[:,j,:]
                    sim[:,:,j] = torch.mul(a.unsqueeze(1),b.unsqueeze(0)).sum(dim=-1)
                sim_sorted,sim_idx = torch.sort(sim,dim=1,descending=True)

                prompt = []
                prompt = []
                for num in range(self.args.topk):
                    idx = sim_idx[:,num,:]
                    prompt_data = []
                    for j in range(idx.shape[0]):
                        data = []
                        for k in range(idx.shape[1]):
                            data.append(dataset[idx[j,k],:,k])
                        data = torch.stack(data,dim=1)
                        prompt_data.append(data)
                    prompt_data = torch.stack(prompt_data,dim=0) 
                    prompt.append(prompt_data)     
                      
                if len(prompt) != 0:
                    prompt = torch.stack(prompt,dim=0).to(self.device)
                else:
                    prompt =None
                
                # select the top k similar retrieval data as prompt
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(prompt, batch_x)
                else:
                    outputs = self.model(prompt, batch_x)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  
                true = batch_y 

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if self.args.visual and i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)
        inputx = np.concatenate(inputx,axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path,exist_ok=True)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        
        mae = torch.tensor(mae).to(self.device)
        mse = torch.tensor(mse).to(self.device)
        rse = torch.tensor(rse).to(self.device)
        if self.args.use_multi_gpu:
            dist.barrier()   
            dist.all_reduce(mae, op=dist.ReduceOp.SUM)
            dist.all_reduce(mse, op=dist.ReduceOp.SUM)
            dist.all_reduce(rse, op=dist.ReduceOp.SUM)
            
            mae = mae.item() / dist.get_world_size()
            mse = mse.item() / dist.get_world_size()
            rse = rse.item() / dist.get_world_size()
        
        if (self.args.use_multi_gpu and self.args.rank == 0) or not self.args.use_multi_gpu:
            print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            f = open("result.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            f.write('\n')
            f.write('\n')
            f.close()
        return
