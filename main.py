from ast import parse
import os
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", type=str)
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--momentum", type=int, default=0.09)
    parser.add_argument("--print_itr", type=int, default=100)
    parser.add_argument("--save_epoch", type=int, default=5)
    parser.add_argument("--log_dir", type=str, default="./log/")
    parser.add_argument("--model_save_path", type=str, default="./checkpoints/")
    parser.add_argument("--onnx_dir", type=str, default="./output/structure/")
    parser.add_argument("--fig_dir", type=str, default="./output/figure")
    parser.add_argument("--load_dir", type=str)
    args = parser.parse_args()
    
    if args.phase == "train":
        import train
        train.train(args)
    elif args.phase == "test":
        import test
        test.test(args)
    else:
        raise ValueError("phase not valid")