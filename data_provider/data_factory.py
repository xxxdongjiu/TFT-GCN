from data_provider.data_loader import PSMSegLoader, MSLSegLoader, SMAPSegLoader, \
    SMDSegLoader, SWATSegLoader, HAISegLoader, CCFDSegLoader, SKABSegLoader
from torch.utils.data import DataLoader

data_dict = {
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'HAI': HAISegLoader,
    'CCFD': CCFDSegLoader,
    'SKAB': SKABSegLoader,
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        batch_size = args.batch_size  # bsz for train and valid

    drop_last = False
    data_set = Data(
        root_path=args.root_path,
        win_size=args.seq_len,
        flag=flag,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
