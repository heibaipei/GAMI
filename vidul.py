import os
import torch.backends.cudnn as cudnn
import torch.utils.data

import argparse
import torch
import os
import sys
from utils.write_csv import csv_file
from utils.dataset import MyDataset
from utils.mainTest import MainTest
from mainModel import MainModel
# from models.deepInfoMaxLoss import DeepInfoMaxLoss
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def MainTest(dataloader, model_root, target_index, device="cuda"):
    cudnn.benchmark = True
    alpha = 0

    """ test """
    # ss = model_root+str(target_index)+"mainModel_best.pth"
    ss = model_root+str(target_index)+"mainModel_best.pth"


    # my_net = torch.load()

    # my_net = torch.load(os.path.join(
    #     model_root, model_name
    # ))
    my_net = torch.load(ss)
    my_net = my_net.eval()

    my_net = my_net.to(device)

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    feature_all = []
    rele_all = []
    irre_all = []
    label_all =[]

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label, _ = data_target

        batch_size = len(t_label)


        t_img = t_img.to(device)
        t_label = t_label.to(device)

        rele,  irre,  feature, class_ouput, atten ,_= my_net(t_img)
        pred = class_ouput.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size
        feature_all.append(feature)
        rele_all.append(rele)
        irre_all.append(irre)
        label_all.append(t_label)

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return feature_all, rele_all, irre_all, label_all


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    target_list = ['6']
    classname = 'valence2'
    datapath = './data'
    model_save_path = '/home/wyd/informer_jsd_deap/checkpoint/test1/'
    data_target = MyDataset(classname, datapath, domain_label=[0], data_load=target_list)
    dataloader_target = DataLoader(dataset=data_target, batch_size=128, shuffle=True, drop_last=True)
    feature_all, rele_all, irre_all, label_all = MainTest(dataloader_target,
                model_save_path, 2, device='cuda:0')
    feature = torch.cat(feature_all, dim=0)
    feature = feature.cpu().detach().numpy()
    rele = torch.cat(rele_all, dim=0).cpu().detach().numpy()
    irre = torch.cat(irre_all, dim=0).cpu().detach().numpy()
    label = torch.cat(label_all, dim=0).cpu().detach().numpy()
    print("feature.shape", feature.shape)
    print("rele.shape", rele.shape)

    np.savez('6.npz', feature=feature, rele=rele, irre = irre, label=label)










