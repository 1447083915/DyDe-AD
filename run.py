import argparse
import torch
import torch.backends
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2026
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='DyDe-AD')

    parser.add_argument('--task_name', type=str, default='anomaly_detection')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='DyDeAD')
    parser.add_argument('--model', type=str, default='Ours')

    parser.add_argument('--data', type=str, default='RflyMAD')
    parser.add_argument('--root_path', type=str, default='./dataset/RflyMAD/')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    parser.add_argument('--seq_len', type=int, default=175)
    parser.add_argument('--enc_in', type=int, default=43)
    parser.add_argument('--c_out', type=int, default=43)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=64)

    parser.add_argument('--anomaly_ratio', type=float, default=0.25)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--lradj', type=str, default='cosine')

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_type', type=str, default='cuda')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    args = parser.parse_args()

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Anomaly_Detection

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = '{}_{}_{}_{}_sl{}_dm{}_ld{}_{}_{}'.format(
                args.task_name, args.model_id, args.model, args.data,
                args.seq_len, args.d_model, args.latent_dim, args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_dm{}_ld{}_{}_{}'.format(
            args.task_name, args.model_id, args.model, args.data,
            args.seq_len, args.d_model, args.latent_dim, args.des, ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
