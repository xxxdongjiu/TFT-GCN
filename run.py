import argparse
import os
import torch
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from utils.print_args import print_args
import random
import numpy as np

if __name__ == "__main__":
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument("--is_training", type=int, required=True, default=1, help="status")

    # data loader
    parser.add_argument("--data", type=str, required=True, default="SWAT", help="dataset type")
    parser.add_argument(
        "--root_path", type=str, default="/home/xusheng/dataset/SWaT/", help="root path of the data file"
    )
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="location of model checkpoints")
    parser.add_argument(
        "--checkpoints_pretrain",
        type=str,
        default="./checkpoints_pretrain/",
        help="location of model checkpoints with pretrain",
    )
    parser.add_argument("--seq_len", type=int, default=100, help="input sequence length")
    parser.add_argument("--anomaly_ratio", type=float, default=0.25, help="prior anomaly ratio (%)")

    # model define
    parser.add_argument("--top_k", type=int, default=5, help="for Block")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--output_attention", action="store_true", help="whether to output attention in encoder")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--n_heads2", type=int, default=4, help="num of heads")
    parser.add_argument("--seg_len", type=int, default=10)
    parser.add_argument("--f_model", type=int, default=8)

    # GNN
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--gdepth", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--hidden", type=int, default=8, help="channel dim")
    parser.add_argument("--nvechidden", type=int, default=8, help="variable vec dim")
    parser.add_argument("--conv_channel", type=int, default=32, help="")
    parser.add_argument("--skip_channel", type=int, default=32, help="")
    parser.add_argument("--node_dim", type=int, default=10, help="each node embbed to dim dimentions")

    # F-block
    parser.add_argument("--fnet_d_model", type=int, default=128, help="dimension of model of fnet")
    parser.add_argument("--fnet_layers", type=int, default=2, help="num of fnet layers")
    parser.add_argument("--fnet_d_ff", type=int, default=1024, help="dimension of fcn of fnet")
    parser.add_argument("--complex_dropout", type=float, default=0.1, help="complex_dropout")
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument("--activation", type=str, default="gelu", help="activation")

    # pretrain
    parser.add_argument("--is_pretraining", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=5)
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--pretrain_epochs", type=int, default=5)

    # optimization
    parser.add_argument("--num_workers", type=int, default=2, help="data loader num workers")
    parser.add_argument("--train_epochs", type=int, default=3, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
    parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision training", default=False)

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple gpus", default=False)
    parser.add_argument("--devices", type=str, default="0,1", help="device ids of multiple gpus")

    args = parser.parse_args()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False
    # print(torch.cuda.is_available())

    # print("Args in experiment:")
    # print_args(args)

    Exp = Exp_Anomaly_Detection

    if args.is_training:
        # setting record of experiments
        exp = Exp(args)  # set experiments

        if args.is_pretraining:
            print(">>>>>>>start pretraining>>>>>>>")
            exp.pretrain()

        setting = "{}_sl{}_dm{}_el{}_df{}".format(args.data, args.seq_len, args.d_model, args.e_layers, args.d_ff)

        print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
        exp.train(setting)

        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
    else:
        setting = "{}_sl{}_dm{}_el{}_df{}".format(args.data, args.seq_len, args.d_model, args.e_layers, args.d_ff)

        exp = Exp(args)  # set experiments
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
