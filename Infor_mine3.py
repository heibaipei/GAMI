import argparse
import torch
import os
import sys
from utils.write_csv import csv_file
from utils.dataset import MyDataset
from utils.mainTest import MainTest
from mainModel import MainModel, estimate_JSD_MI, CLUBSample, Global_disc_EEGNet,Local_disc_EEGNet, MINE
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(6666)


## run the dataset of Deap
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_model', type=int, default=64) # a
    parser.add_argument('--n_heads', type=int, default=2) #
    parser.add_argument('--e_layers', type=int, default=5) # b
    parser.add_argument('--length', type = int, default= 8) # final length
    parser.add_argument('--nfeatr', type = int, default=64*8) # a* 128/2^b

    parser.add_argument('--convlinesize', type=int, default =512)
    parser.add_argument('--convoutsize', type=int, default = 128)
    parser.add_argument('--weight1', type = int, default = 225)

    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--indim', type=int, default=128)
    parser.add_argument('--hiddim', type=int, default=64)

    parser.add_argument('--err_w', type=int, default=1)
    parser.add_argument('--dim_w', type=int, default=1)

    parser.add_argument('--data_load', type=list, default=list(range(1, 15)))
    # parser.add_argument('--data_load', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--single', type=int, default=23)
    parser.add_argument('--target_index', type=int, default=3)
    parser.add_argument('--classname', type=str, default='arousal2')
    parser.add_argument('--nbClasses', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--vae_lr', type=float, default=1e-1)
    parser.add_argument('--dann_lr', type=float, default=1e-1)
    parser.add_argument('--datapath', type=str, default='./data')
    parser.add_argument('--savepath', type=str, default='/home/wyd/informer_jsd_deap/checkpoint')
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))#
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    print('============ Start =============')
    args = config()
    args.model_save_path = f'{args.savepath}/test1/'
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    # csv_result = csv_file(args.savepath+f"/result/muilty"+str(args.target_index), "/result.csv", ["epoch", "accu_t"])

    save_name = args.savepath+f"/result3/"+str(args.target_index)
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    name = args.savepath + f"/result3/" + str(args.target_index) + "/result.npz"

    data_load = args.data_load
    target_list = []  # 预测的数据
    target_list.append(data_load[args.target_index])
    del data_load[args.target_index]
    print(data_load)
    print(target_list)
    train_data = MyDataset(args.classname, args.datapath, domain_label=list(range(1, len(data_load) + 1)),
                           data_load=data_load)
    data_target = MyDataset(args.classname, args.datapath, domain_label=[0], data_load=target_list)

    dataloader_source = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataloader_target = DataLoader(dataset=data_target, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print("the data have load")

    # 数据加载
    # data_load = args.data_load
    # # print("data_load", data_load)
    # Alldata = MyDataset(args.classname, args.datapath, domain_label=list(range(1, len(data_load) + 1)), data_load=data_load)
    # # Alldataload = DataLoader(dataset=Alldata, batch_size=args.batch_size, shuffle=True)
    # train_size = int(0.9 * len(Alldata))
    # test_size = len(Alldata) - train_size
    # print("train_size, test_size", train_size, test_size)
    #
    # train_data, data_target = torch.utils.data.random_split(Alldata, [train_size, test_size])
    # dataloader_source = DataLoader(dataset=train_data, batch_size=args.batch_size,
    # 	shuffle=True, drop_last=True)
    # dataloader_target = DataLoader(dataset=data_target, batch_size=args.batch_size,
    # 	shuffle=True,drop_last=True)


    # 声明模型
    mainModel = MainModel(c_in=1, d_model=args.d_model, n_heads=args.n_heads,
    	e_layers=args.e_layers, convlinesize=args.convlinesize, 
    	convoutsize=args.convoutsize, out_l=len(data_load)+1).to(args.device)
    print("Claim the model is over")

    nfeatr = 512
    nfeati = 512
    nfeatl = args.d_model
    # nfeatl = 180  ##
    num_ch = 1
    nfeatg = 32
    nfeatl2 = 32

    
    club = CLUBSample(nfeatr, nfeati, 128).to(args.device)
    mine = MINE(nfeatr, nfeati).to(args.device)

    local_disc = Local_disc_EEGNet(nfeatl=nfeatl, nfeatg=nfeatg, nfeatl2=nfeatl2, num_ch=num_ch).to(args.device)
    global_disc = Global_disc_EEGNet(nfeatl=nfeatl, nfeatg=nfeatg, num_ch=num_ch).to(args.device)

    optimizer = optim.Adam(mainModel.parameters(), lr=args.vae_lr)
    optimizer1 = optim.Adam(global_disc.parameters(), lr=args.dann_lr)
    optimizer2 = optim.Adam(local_disc.parameters(), lr=args.dann_lr)
    # optimizer3 = optim.Adam(mine.parameters(), lr=args.dann_lr)
    optimizer4 = optim.Adam(club.parameters(), lr=args.dann_lr)
    loss_class = nn.CrossEntropyLoss()
    loss_class = loss_class.to(args.device)
    loss_domain = nn.CrossEntropyLoss()
    loss_domain = loss_domain.to(args.device)
    # print("迭代器声明完成!")

    # training

    best_accu_t = 0.0
    last_s = 0
    last_t = 0
    mainModel.train()
    # mine.train()
    local_disc.train()
    global_disc.train()
    club.eval()
    for epoch in range(args.epochs):
        len_dataloader = len(dataloader_source)
        print(len_dataloader)
        data_source_iter = iter(dataloader_source)
        # data_target_iter = iter(dataloader_target)
        i = 0
        n_total = 0
        n_total2 = 0
        n_correct = 0
        n_correct_s = 0
        n_correct_t = 0
        for i in range(len_dataloader):
            # training model using source data
            data_source = data_source_iter.next()
            s_data, s_label, s_domain_label = data_source
            # 数据加载进cuda
            s_data = s_data.to(args.device)
            s_label = s_label.to(args.device)
            s_domain_label = s_domain_label.to(args.device)

            # training model using target data
            # data_target = data_target_iter.next()
            # t_data, t_label, t_domain_label = data_target
            # # 数据加载进cuda
            # t_data = t_data.to(args.device)
            # t_label = t_label.to(args.device)
            # t_domain_label = t_domain_label.to(args.device)

            #train
            mainModel.zero_grad()
            optimizer.zero_grad()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # optimizer3.zero_grad()
            optimizer4.zero_grad()

            # feature_s, class_ouput_s, domain_output_s, _ = mainModel(s_data)


            rele, irre, feature, class_ouput_s, atten, domain_output_s = mainModel(s_data)
            # _, _, _, class_ouput_t, atten_t, t_d_label = mainModel(t_data)

            # To ensure good decomposition, estimate MI between relevant feature and irrelevant feature
            rele_ = torch.reshape(rele, (rele.shape[0], -1))  # [batch, d1*t1]
            irre_ = torch.reshape(irre, (irre.shape[0], -1))  # [batch, d1*t1]

            # print("rele.shape", rele.shape)
            # print("irre.shape", irre.shape)
            # ishuffle = torch.index_select(irre_, 0, torch.randperm(irre_.shape[0]).to((args.device)))
            # djoint = mine(rele_, irre_)  # [batch, 1]
            # dmarginal = mine(rele_, ishuffle)  # [batch, 1]
            # loss_decomposition2 = -estimate_JSD_MI(djoint, dmarginal, True)
            loss_decomposition2 = club(rele_, irre_)

            # Estimate global MI
            globalf = feature  #64*128
            gshuffle = torch.index_select(globalf, 0, torch.randperm(globalf.shape[0]).to(args.device))  # [batch, d2]
            gjoint = global_disc(rele, globalf)  # [batch, 1]
            gmarginal = global_disc(rele, gshuffle)  # [batch, 1]
            loss_global_mi = -estimate_JSD_MI(gjoint, gmarginal, True)

            # Estimate local MI
            ljoint = local_disc(rele, globalf)
            lmarginal = local_disc(rele, gshuffle)
            loss_local_mi = -estimate_JSD_MI(ljoint, lmarginal, True)
            # loss_local_mi = temp.mean()
            #+ loss_local_mi

            loss_mi = (loss_local_mi + loss_global_mi)

            err_s_label = loss_class(class_ouput_s, s_label)
            err_s_domain = loss_domain(domain_output_s, s_domain_label)

            # err_t_domain + err_s_domain
            # err = err_s_label*0.2 + loss_decomposition*0.5 + loss_dim *0.1+ loss_decomposition2 + loss_dim *0
            err = (err_s_label + err_s_domain)*10+ (loss_decomposition2)*0.3
            loss = args.err_w * err

            pred_s = class_ouput_s.data.max(1, keepdim=True)[1]
            # pred_t = class_ouput_t.data.max(1, keepdim=True)[1]

            # n_correct_t += pred_t.eq(t_label.data.view_as(pred_t)).cpu().sum()
            n_correct_s += pred_s.eq(s_label.data.view_as(pred_s)).cpu().sum()
            n_total += args.batch_size
            i += 1



            loss.backward()
            optimizer.step() # whole model
            optimizer1.step()  # local
            optimizer2.step()  # global
            # optimizer3.step()   # Mine
            for j in range(5):
                club.train()
                mi_loss = club.learning_loss(rele_.detach(), irre_.detach())
                club.zero_grad()
                mi_loss.backward()
                optimizer4.step()



            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], '
                             'err_s_label: %f,'
                             'loss_decomposition: %f, '
                             'loss_global_mi: %f,'
                             'loss_local_mi: %f,'
                             'err_s_domain: %f.'\
                             % (epoch, i + 1, len_dataloader,
                                err_s_label.data.cpu().numpy(),
                                loss_decomposition2.data.cpu().numpy(),
                                loss_global_mi.data.cpu().numpy(),
                                loss_local_mi.data.cpu().numpy(),
                                err_s_domain.data.cpu().numpy()))

            sys.stdout.flush()

        # accu_t = n_correct_t.data.numpy() * 1.0 / n_total
        accu_s = n_correct_s.data.numpy() * 1.0/ n_total

        print('\n')
        print("epoch", epoch)
        # print("accu_t", accu_t)
        print("accu_s", accu_s)
        # torch.save(mainModel, args.model_save_path + 'mainModel.pth')
        # accu_s = MainTest(dataloader_source, args.model_save_path,'mainModel.pth', device=args.device)
        # print('Accuracy of the %s dataset: %f' % ('source', accu_s))
        # accu_t = MainTest(dataloader_target, args.model_save_path, 'mainModel.pth', device=args.device)
        # csv_result.write_a(epoch = epoch, accu_t = accu_t)
        # print('Accuracy of the %s dataset: %f' % ('target', accu_t))

        mainModel.eval()
        data_target_iter = iter(dataloader_target)

        for j in range(len(dataloader_target)):
            data_target = data_target_iter.next()
            t_data, t_label, t_domain_label = data_target
            # 数据加载进cuda
            t_data = t_data.to(args.device)
            t_domain_label = t_domain_label.to(args.device)
            t_label = t_label.to(args.device)
            rele, irre, feature, class_ouput_t, atten, domain_output_s = mainModel(t_data)
            pred_t = class_ouput_t.data.max(1, keepdim=True)[1]
            n_correct_t += pred_t.eq(t_label.data.view_as(pred_t)).cpu().sum()
            n_total2 += args.batch_size
        accu_t = n_correct_t.data.numpy() * 1.0 / n_total2
        print("accu_t", accu_t)


        if accu_t > best_accu_t:
            best_accu_t = accu_t
            torch.save(mainModel, args.model_save_path + str(args.target_index)+'mainModel_best.pth')
        print("the best result", best_accu_t)
        if accu_s >last_s:
            last_t = accu_t
            last_s = accu_s
    print("last_t", last_t, "accu_s", accu_s)

    import numpy as np
    np.savez(name, best_accu_t=best_accu_t, accu_s=last_s, accu_t=last_t)
