import os
import time
import copy

import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.autograd import Variable

import select_model
import dataloader
import test


def get_model(args, net):
    train_model = net.se_resnet_18()
    if args.is_transfer:
        try:
            transfer_models = select_model.select_model(args.save_model_path)
            print('Please select the saved model you want to transfer:\n{}'.format(transfer_models))
            model_index = int(input())
            checkpoint = torch.load(os.path.join(args.save_model_path).replace("\\", "/")
                              + '/' + '{}'.format(transfer_models[model_index]))
            pre_param = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.state_dict().items())}
            train_model.load_state_dic(pre_param)
            print('Load model {} successfully.'.format(transfer_models[model_index]))
            return train_model
        except Exception as err:
            print('Error : model load ----- >>', err)
    else:
        return train_model


def train(args):
    try:

        """
        ['se_resnet_18','se_resnet_34','se_resnet_50','se_resnet_101','se_resnet_152',]
        """
        # ---------------------    Save the training results to excel    -----------------------
        path = os.path.join(args.save_result_path, args.model_name).replace("\\", "/")
        if not os.path.exists(path):
            os.makedirs(path)

        best_model_acc = 0
        best_matrix = None
        best_model = copy.deepcopy(model)
        Writer = pd.ExcelWriter('{}/train_{}_cross_validation.xlsx'.format(path, args.k))
        Matrix = pd.ExcelWriter('{}/test_{}_cross_validation.xlsx'.format(path, args.k))

        for i in range(args.k):
            try:
                net = importlib.import_module("models.{}".format(args.model_name.split(".")[0]))
                model = get_model(args=args,net=net).to(args.device)
                start_time = time.time()
                # ----------------------------     Load dataset    -------------------------------------
                train_dataset, train_dataloader, train_datasize = dataloader.get_data(args,i,flag = 'train')                
                test_dataset, test_dataloader, test_datasize = dataloader.get_data(args,i,flag = 'test')

                print('[Training data]:[{}]\n[Testing data]:[{}]'.format(train_datasize, test_datasize))

                # -------------------------     Model Initialization    --------------------------------
                model.train()
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.95) 
                #change lr to gamma*lr each step_size epochs

                # ---------------------------     Steps computaion    ----------------------------------
                total_steps = len(train_dataloader)  # to print the process obviously
                total_trained_samples = 0  # to calculate the average of accuracy
                total_right_pred = 0
                total_loss = 0
                epochs = []
                learning_rates = []
                train_losses = []
                train_accuracies = []
                test_accuracies = []
                for epoch in range(args.epochs):
                    for step, (batch_image, batch_label) in enumerate(train_dataloader):
                        try:
                            inputs = Variable(batch_image.to(args.device))
                            labels = Variable(batch_label.to(args.device))

                            total_trained_samples += labels.size(0)
                            optimizer.zero_grad()  # param initialization

                            outputs = model(inputs)
                            _, prediction = torch.max(outputs.data, 1)
                            step_loss = criterion(outputs, labels)
                            acc = total_right_pred/total_trained_samples
                            print('Train:[{}/{}]-Epoch:[{}/{}]-Lr[{:.4e}]-Step:[{}/{}]--Loss:[{:.2%}]--Acc[{:.2%}]'.
                                  format(i, args.k, epoch, args.epochs, lr_scheduler.get_lr()[0],
                                         step, total_steps, step_loss.item(), acc))
                            total_right_pred += (prediction == labels).sum().item()
                            total_loss += step_loss.item()

                            step_loss.backward()
                            optimizer.step()
                        except Exception as err:
                            print('Error : train step  ----- >>', err)
                            print("Error : train step  ---- >>", err.__traceback__.tb_lineno)

                    test_acc, statistic_matrix = test.test(args,
                                                           test_dataloader,
                                                           model, epoch, i)
                    if test_acc > best_model_acc:
                        best_model_acc = test_acc
                        best_model = copy.deepcopy(model.state_dict())
                        best_matrix = statistic_matrix

                    epochs.append(epoch)
                    learning_rates.append(lr_scheduler.get_lr()[0])
                    train_losses.append(total_loss / total_trained_samples)
                    train_accuracies.append(total_right_pred / total_trained_samples)
                    test_accuracies.append(test_acc)
                    lr_scheduler.step()
                    model.to(args.device)
                    model.train()
                train_epoch_pd = pd.DataFrame({"epoch": epochs, 'learning_rate': learning_rates,
                                               "train_loss": train_losses, "train_acc": train_accuracies,
                                               "test_acc": test_accuracies})
                train_epoch_pd.to_excel(Writer, sheet_name='folder_{}'.format(i))

                test_matrix = pd.DataFrame(best_matrix)
                test_matrix.to_excel(Matrix, sheet_name='folder_{}_{}'.format(i, best_model_acc))

                Writer.save()
                Matrix.save()
                torch.save(model, '{}/{}_{}_{}.pth'.
                           format(args.save_model_path, args.model_name, args.k, i))
            except Exception as err:
                print("Error:train epoch  ---- >>", err)
                print("Error:train epoch  ---- >>", err.__traceback__.tb_lineno)
    except Exception as err:
        print("Error:train folder  ---- >>", err)
        print("Error:train folder ---- >>", err.__traceback__.tb_lineno)

