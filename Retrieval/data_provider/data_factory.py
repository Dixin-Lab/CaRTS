from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'electricity': Dataset_Custom,
    'illness': Dataset_Custom,
    'weather': Dataset_Custom,
    'exchange': Dataset_Custom,
    'traffic': Dataset_Custom
}


def data_provider(args, flag):
    is_retrieval = False
    if flag == 'retrieval':
        data = args.retrieval_data
        root_path = args.retrieval_root_path
        data_path = args.retrieval_data_path
        drop_last = False
        flag = 'train'
        is_retrieval = True
    else:
        data = args.data
        root_path = args.root_path
        data_path = args.data_path
    Data = data_dict[data]
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    if is_retrieval:
        drop_last = False
    data_set = Data(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len]
    )
    
    if is_retrieval:
        flag = 'retrieval'

    if (args.use_multi_gpu and args.rank == 0) or not args.use_multi_gpu:
        print(flag, len(data_set))
    
    if args.use_multi_gpu:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(data_set, 
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last,
            )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
    return data_set, data_loader
