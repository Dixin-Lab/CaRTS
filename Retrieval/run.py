import argparse
import os
import sys
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import torch.distributed as dist
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CaRTS')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--is_pretraining', type=int, default=0, help='pretraining status')
    parser.add_argument('--is_plot', type=int, default=0, help='plot status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Retrieval', help='model name')
    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file')
    parser.add_argument('--retrieval_data', type=str, default='ETTm2', help='retrieval dataset type')
    parser.add_argument('--retrieval_root_path', type=str, default='./dataset/', help='root path of the retrieval file')
    parser.add_argument('--retrieval_data_path', type=str, default='./ETTm2.csv', help='retrieval file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--head_dropout', type=float, default=0, help='head dropout')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
    parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, default=0, help='rank of distributed processes')

    
    parser.add_argument('--topk',type=int,default=1,help='for retrieval topk')
    parser.add_argument('--normalization',type=bool,default=False,help='for similarity normalization')
    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.use_multi_gpu:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        else:
            print('Not using distributed mode')
        torch.cuda.set_device(args.gpu)

        args.dist_url = 'env://'  
        args.dist_backend = 'nccl' 
        print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        dist.barrier() 

    if (args.use_multi_gpu and args.rank == 0) or not args.use_multi_gpu:
        print('Args in experiment:')
        print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_df{}_wd{}_lr{}_norm{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.retrieval_data,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_ff,
                args.weight_decay,
                args.learning_rate,
                args.normalization,
                args.des,ii)
            
            if args.use_multi_gpu:
                setting = setting + "_DDP"
                
            exp = Exp(args)  # set experiments
            if (args.use_multi_gpu and args.rank == 0) or not args.use_multi_gpu:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            if (args.use_multi_gpu and args.rank == 0) or not args.use_multi_gpu:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            
            torch.cuda.empty_cache()
            if args.use_multi_gpu:
                dist.barrier()
                dist.destroy_process_group()
            
    elif args.is_pretraining:  
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_df{}_wd{}_lr{}_norm{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.retrieval_data,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_ff,
                args.weight_decay,
                args.learning_rate,
                args.normalization,
                args.des,ii)
        if args.use_multi_gpu:
            setting = setting + "_DDP"
        exp = Exp(args)  # set experiments
        if (args.use_multi_gpu and args.rank == 0) or not args.use_multi_gpu:
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train_pretrain(setting)
        torch.cuda.empty_cache()
        if args.use_multi_gpu:
            dist.barrier()
            dist.destroy_process_group()
    elif args.is_plot:
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_df{}_wd{}_lr{}_norm{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.retrieval_data,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_ff,
                args.weight_decay,
                args.learning_rate,
                args.normalization,
                args.des,ii)
        if "ETTm1" in args.data or 'weather' in args.data:
            setting = setting + "_DDP"
        
        exp = Exp(args)  # set experiments
        if (args.use_multi_gpu and args.rank == 0) or not args.use_multi_gpu:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.plot_sim(setting)
        torch.cuda.empty_cache()
        if args.use_multi_gpu:
            dist.barrier()
            dist.destroy_process_group()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_df{}_wd{}_lr{}_norm{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.retrieval_data,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_ff,
                args.weight_decay,
                args.learning_rate,
                args.normalization,
                args.des,ii)
        if args.use_multi_gpu:
            setting = setting + "_DDP"
        
        exp = Exp(args)  # set experiments
        if (args.use_multi_gpu and args.rank == 0) or not args.use_multi_gpu:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
        if args.use_multi_gpu:
            dist.barrier()
            dist.destroy_process_group()
        