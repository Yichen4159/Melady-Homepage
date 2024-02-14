from data_provider.data_loader import Dataset_Custom, Dataset_Pred, Dataset_TSF, Dataset_ETT_hour, Dataset_ETT_minute
from torch.utils.data import DataLoader

data_dict = {
    'Weather': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
}

data_path_dict = {
    'ETTm1': 'ETTm1.csv',
    'ETTm2': 'ETTm2.csv',
    'ETTh1': 'ETTh1.csv',
    'ETTh2': 'ETTh2.csv',
    'Weather': 'weather.csv',
    'Traffic': 'traffic.csv'
}

def get_test_data_loader(data_name, pred_len):
    Data = data_dict[data_name]
    timeenc = 0
    percent = 100
    max_len = -1
    flag='test'
    shuffle_flag = False
    drop_last = True
    batch_size = 1
    freq = 'h'
    seq_len = 336
    label_len = 168
    num_workers=0
    
    data_set = Data(
        root_path='dataset',
        data_path=data_path_dict[data_name],
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features='M',
        target='OT',
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        max_len=max_len,
        train_all=False,
        data_name = data_name
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last)
    return data_set