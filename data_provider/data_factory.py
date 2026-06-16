from data_provider.data_loader import UAVSegLoader
from torch.utils.data import DataLoader

data_dict = {
    'RflyMAD': UAVSegLoader,
    'ALFA': UAVSegLoader,
    'UAV_RFD': UAVSegLoader,
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size

    drop_last = False
    data_set = Data(
        args=args,
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
        drop_last=drop_last,
    )
    return data_set, data_loader
