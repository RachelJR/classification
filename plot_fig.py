"""
according to the results to plot
1.AUC,ROC curves of each class
2.train loss-acc and test acc curves
3.visualize the statistic matrix through a figure
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import pandas as pd
import seaborn as sns
import numpy as np
from pylab import *


def plot_single_model_curves(file):
    sheet_names = pd.ExcelFile(file).sheet_names
    save_path = os.path.dirname(file)
    for sheet in sheet_names:
        data = pd.read_excel(file,sheet_name = sheet)
        folder = sheet.split('_')[1]
        data.plot(x = 'epoch',y=['train_acc','train_loss','test_acc'], colors = ['DarkGreen','DarkBlue','Red'],
                  sharex = True,sharey = True,figsize = (10,8))
        plt.title('4-Cross-validation Training : 0{}'.format(int(folder)+1))
        plt.legend(loc='upper right',bbox_to_anchor=(1, 0.5))
        plt.xticks(np.arange(0,150,10))
        plt.yticks(np.arange(0,1,0.1))
        plt.grid()
        plt.xlabel('Epochs')
        plt.ylabel('Rate')
        plt.savefig('{}/train_{}.png'.format(save_path,folder))

def plot_single_model_fused_curves(file):
    sheet_names = pd.ExcelFile(file).sheet_names
    save_path = os.path.dirname(file)
    data0 = pd.read_excel(file,sheet_name =  sheet_names[0])
    data1 = pd.read_excel(file,sheet_name =  sheet_names[1])
    data2 = pd.read_excel(file,sheet_name =  sheet_names[2])
    data3 = pd.read_excel(file,sheet_name =  sheet_names[3])
    x_axis = data0['epoch']
    sub_axis = filter(lambda x:x%200 == 0,x_axis)
    fig = plt.figure(figsize=(10,8))
    plt.title("Model {} training curves".format(save_path.split('/')[-1]))
    l1 = plt.plot(x_axis,np.transpose(
        [data0['train_acc'],data1['train_acc'],data2['train_acc'],data3['train_acc']]),'b-')
    l2 = plt.plot(x_axis,np.transpose(
        [data0['test_acc'],data1['test_acc'],data2['test_acc'],data3['test_acc']]),'r-')
    l3 = plt.plot(x_axis,np.transpose(
        [data0['train_loss'],data1['train_loss'],data2['train_loss'],data3['train_loss']]),'g-')
    blue = mpatch.Patch(color='blue', label='train acc')
    red = mpatch.Patch(color='red', label='test loss')
    green = mpatch.Patch(color='green', label='train loss')
    plt.legend(handles = [blue,red,green],loc = (1,1), bbox_to_anchor=(1,0))
    plt.xticks(np.arange(0,150,10))
    plt.yticks(np.arange(0,1,0.1))
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.savefig('{}/fuse.png'.format(save_path))
    
if __name__ == '__main__':
    # training results
    train_file = '/home/python/code/classification/saved_results/GLSE_net/train_4_cross_validation.xlsx'
    #testing results
    test_pd =  pd.read_excel(
        '/home/python/code/classification/saved_results/glse_gate_net/test_4_cross_validation.xlsx',None)
    plot_single_model_curves(train_file)
    plot_single_model_fused_curves(train_file)