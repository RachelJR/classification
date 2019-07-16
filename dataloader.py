from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pylab import *
from PIL import Image

import os
import pydicom
import cv2
import numpy as np
import pandas as pd

inter_dic = {'nearest': 'cv2.INTER_NEAREST',
             'linear': cv2.INTER_LINEAR,
             'area': cv2.INTER_AREA,
             'cubic': 'cv2.INTER_CUBIC',
             'lanczos4': cv2.INTER_LANCZOS4,
             'linear_exact': cv2.INTER_LINEAR_EXACT,}
def write_excel(args):
    try:
        # dcm/train,val
        Writer = pd.ExcelWriter('{}/data.xlsx'
                                .format(os.path.join(args.data_path,args.data_type).replace("\\","/")))
        imgs = []
        labels = []
        flags = [os.path.join(args.data_path,args.data_type,flag).replace("\\","/") 
                 for flag in os.listdir(os.path.join(args.data_path,args.data_type).replace("\\","/")) 
                     if os.path.isdir(os.path.join(args.data_path,args.data_type,flag).replace("\\","/"))]
        for flag in flags:
            for label in os.listdir(flag):
                for img in os.listdir(os.path.join(flag,label).replace("\\","/")):
                    imgs.append("{}/{}/{}".format(flag,label,img)) 
                    labels.append(label)
        dt = pd.DataFrame({"img":imgs,"label":labels})
        dt.to_excel(Writer)
        Writer.save()  
        print("All data has been wirtten to the \n[{}]".format(Writer.path))
        return Writer.path
    except Exception as err:
        print('Error : write excel ----- >>', err)
        print("Error : write excel  ---- >>", err.__traceback__.tb_lineno)
        
def generate_txt(args,excel):
    try:
        data = pd.read_excel(excel)
        indexes = np.arange(0,len(data),1)
        for i in range(args.k):
            train_txt = open('{}/{}/train_{}_{}.txt'.format(args.data_path,args.data_type,args.k,i),'w')
            test_txt = open('{}/{}/test_{}_{}.txt'.format(args.data_path,args.data_type,args.k,i),'w')
            try:
                test_indexes = np.random.choice(indexes,size=int(len(data)/args.k))
                for index in indexes:
                    [img,label] = data.loc[index,["img","label"]]
                    if index not in test_indexes:                
                        train_txt.write("{} {}\n".format(img,label))
                    else:
                        test_txt.write("{} {}\n".format(img,label)) 
                
            except Exception as err:
                print('Error : generate {} txt ----- >>'.format(i), err)
                print("Error : generate {} txt  ---- >>".format(i), err.__traceback__.tb_lineno)
    except Exception as err:
        print('Error : generate txt ----- >>', err)
        print("Error : generate txt  ---- >>", err.__traceback__.tb_lineno) 
        
def read_img(img):
    try:
        if img.endswith("dcm"):
            arr = pydicom.read_file(img)
            arr = arr.pixel_array
        else:
            arr = Image.open(img)
        return np.array(arr,dtype='float32')
    except Exception as err:
        print('Error : read image ----- >>', err)
        print("Error : read image  ---- >>", err.__traceback__.tb_lineno)

def data_resize(arr):
    try:
        resized_arr = cv2.resize(arr, (data_shape, data_shape), interpolation=cv2.INTER_CUBIC)
        return resized_arr
    except Exception as err:
        print('Error : data resize ----- >>', err)
        print("Error : data resize  ---- >>", err.__traceback__.tb_lineno)


def data_uniform(img):
    try:
        arr = read_img(img)
        arr = np.add(arr, -1024)
        arr = np.clip(arr, -100, 400)
        arr = data_resize(arr)
        return arr
    except Exception as err:
        print('Error : data uniform ----- >>', err)
        print("Error : data uniform  ---- >>", err.__traceback__.tb_lineno)


class GetDatasets(Dataset):
    def __init__(self, file, transform=None, target_transform=None):
        fh = open(file, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            data = line.split()
            imgs.append((data[0], int(data[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, label = self.imgs[index]
        img = data_uniform(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def get_data(args, i, flag = 'train'):    
    txt_path = [os.path.join(args.data_path + '/' + args.data_type + '/' + files).replace("\\", "/")
                            for files in os.listdir(os.path.join(args.data_path + '/' + args.data_type).replace("\\", "/"))
                            if files == "{}_{}_{}.txt".format(flag,args.k, i)]

    if txt_path == []:
        print("TXT file is none\n Starting generating")
        excel = write_excel(args)
        generate_txt(args,excel)
        [txt] = [os.path.join(args.data_path + '/' + args.data_type + '/' + files).replace("\\", "/")
                            for files in os.listdir(os.path.join(args.data_path + '/' + args.data_type).replace("\\", "/"))
                            if files == "{}_{}_{}.txt".format(flag,args.k, i)]
    else:
        [txt] = txt_path

    global data_shape, data_process
    data_shape = args.data_shape
    data_process = args.data_process

    if flag == 'train':
        shuffle = True
    else:
        shuffle = False
    try:
        data_transform = transforms.Compose([transforms.ToTensor()])
        dataset = GetDatasets(txt, transform=data_transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)
        dataset_size = dataset.__len__()

        return dataset, dataloader, dataset_size
    except Exception as err:
        print("Error:get data  ---- >>", err)
        print("Error:get data  ---- >>", err.__traceback__.tb_lineno)

