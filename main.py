import argparse
import sys
import os
import re
import torch

import select_model
import models
import dataloader
import train
import test
import evaluators
import plot_fig

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace("\\","/")+'/'+'models')
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace("\\","/")+'/'+'datasets')
model_list = []


def check_path(path):
    if not os.path.exists(path):
        print("PATH [{}] is not exist".format(path))
    else:
        print('Check path successfully')


def make_path(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def set_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    
    print("Please add order...")
    print("please input k folder k is:")
    k = int(input())
    print("please choose the model you want:")
    model_list = select_model.select_model(model_path)
    print("-" * 50 + '\n{}'.format(model_list) + '\n' + '-' * 50)
    model_id = int(input())
    print("The model [{}] will be trained with [{}] folder cross-validation".format(model_list[model_id], int(k)))
    parser = argparse.ArgumentParser(description="Training classification network for liver lesions:HCC,MET,HEALTH,HEM ")
    parser.add_argument('--k', type=int, default=k)
    parser.add_argument('--model_path', type=str, default=model_path)

    parser.add_argument('--model_name', type=str, default=model_list[model_id])

    parser.add_argument('--data_path', type=str, default=data_path)
    parser.add_argument('--data_type', type=str, default='dcm')
    parser.add_argument('--data_process', type=str, default='cubic')
    parser.add_argument('--data_shape', type=int, default=64)
    parser.add_argument('--data_transform', type=bool, default=False)

    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--is_transfer', type=bool, default=False)

    parser.add_argument('--num_class', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.02)

    parser.add_argument('--save_model_path', type=str, default= os.path.join(
        os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+'/'+'saved_models'))
    parser.add_argument('--save_result_path', type=str, default= os.path.join(
        os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+'/'+'saved_results'))

    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    args.device = set_device()

    print('Check model path:', args.model_path)
    check_path(args.model_path)
    print('Check data path:', args.data_path)
    check_path(args.data_path)

    make_path(args.save_model_path)
    make_path(args.save_result_path)

    if args.is_train:
        train.train(args)
    else:
        test(args)

