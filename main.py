import argparse
import torch
import os
import sys
from utils.write_csv import csv_file
from utils.dataset import MyDataset
from utils.mainTest import MainTest
from mainModel import MainModel
from models.deepInfoMaxLoss import DeepInfoMaxLoss
from torch.utils.data import DataLoader
import torch.optim as optim


def config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--d_model', type=int, default=160)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--e_layers', type=int, default=3)

    parser.add_argument('--convlinesize', type=int, default=64*7*39)
    parser.add_argument('--convoutsize', type=int, default=64)

    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--indim', type=int, default=128)
    parser.add_argument('--hiddim', type=int, default=64)

    parser.add_argument('--err_w', type=int, default=1)
    parser.add_argument('--dim_w', type=int, default=1)

    # parser.add_argument('--data_load', type=list, default=list(range(1, 33)))
    parser.add_argument('--data_load', type=list, default=[1, 2, 3, 4, 5, 6, 8])
    parser.add_argument('--target_index', type=int, default=0)
    parser.add_argument('--classname', type=str, default='arousal2')
    parser.add_argument('--nbClasses', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--vae_lr', type=float, default=1e-5)
    parser.add_argument('--dann_lr', type=float, default=1e-5)
    parser.add_argument('--datapath', type=str, default='/home/wc/code/data/wc_deep_data')
    parser.add_argument('--savepath', type=str, default='/home/wc/code/informer_jsp/checkpoint')
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))#
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    print('============ Start =============')
    args = config()
    args.model_save_path = f'{args.savepath}/main/{args.target_index}/'
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    csv_result = csv_file(args.savepath+f"/result/muilty/target{args.target_index}", "/result.csv", ["epoch", "accu_s", "accu_t", "err_s_label", "err_s_domain", "err_t_domain" ])

    # ????????????
    # data_load = args.data_load
    # Alldata = MyDataset(args.classname, args.datapath, domain_label=list(range(1, len(data_load) + 1)), data_load=data_load)
    # # Alldataload = DataLoader(dataset=Alldata, batch_size=args.batch_size, shuffle=True)
    # train_size = int(0.8 * len(Alldata))
    # test_size = len(Alldata) - train_size
    # print(train_size, test_size)
    #
    # train_data, data_target = torch.utils.data.random_split(Alldata, [train_size, test_size])
    #
    # dataloader_source = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    # dataloader_target = DataLoader(dataset=data_target, batch_size=args.batch_size, shuffle=True)


    data_load = args.data_load
    target_list = []#???????????????
    target_list.append(data_load[args.target_index])
    del data_load[args.target_index]
    print(data_load)
    print(target_list)

    train_data = MyDataset(args.classname,  args.datapath, domain_label=list(range(1, len(data_load)+1)), data_load=data_load)
    data_target = MyDataset(args.classname,  args.datapath, domain_label=[0], data_load=target_list)

    dataloader_source = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    dataloader_target = DataLoader(dataset=data_target, batch_size=args.batch_size, shuffle=True)

    # ????????????
    mainModel = MainModel(d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, convlinesize=args.convlinesize, convoutsize=args.convoutsize, out_l=len(data_load)+1).to(args.device)
    deepInfoMaxLoss = DeepInfoMaxLoss(alpha=args.alpha, indim=args.indim, hiddim=args.hiddim).to(args.device)
    print("??????????????????!")

    optimizer = optim.Adam(mainModel.parameters(), lr=args.dann_lr)
    deepInfoMaxLossOpt = optim.Adam(deepInfoMaxLoss.parameters(), lr=args.dann_lr)
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()
    loss_class = loss_class.to(args.device)
    loss_domain = loss_domain.to(args.device)
    print("?????????????????????!")

    # training
    best_accu_t = 0.0
    mainModel.train()
    deepInfoMaxLoss.train()
    for epoch in range(args.epochs):

        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)
        for i in range(len_dataloader):
            # training model using source data
            data_source = data_source_iter.next()
            s_data, s_label, s_domain_label = data_source
            # ???????????????cuda
            s_data = s_data.to(args.device)
            s_label = s_label.to(args.device)
            s_domain_label = s_domain_label.to(args.device)

            # training model using target data
            data_target = data_target_iter.next()
            t_data, _, t_domain_label = data_target
            # ???????????????cuda
            t_data = t_data.to(args.device)
            t_domain_label = t_domain_label.to(args.device)

            #train
            mainModel.zero_grad()
            deepInfoMaxLoss.zero_grad()
            optimizer.zero_grad()
            deepInfoMaxLossOpt.zero_grad()

            feature_s, class_ouput_s, domain_output_s, _ = mainModel(s_data)

            feature_t, _, domain_output_t, _ = mainModel(t_data)

            err_s_label = loss_class(class_ouput_s, s_label)
            err_s_domain = loss_domain(domain_output_s, s_domain_label)

            err_t_domain = loss_domain(domain_output_t, t_domain_label)
            # err_t_domain + err_s_domain +
            err = err_s_label + err_t_domain + err_s_domain



            if(feature_t.shape==feature_s.shape):
                M_prime = torch.cat((feature_s[1:], feature_s[0].unsqueeze(0)), dim=0)
                dim_l = deepInfoMaxLoss(feature_s, feature_t, M_prime)
            else:
                feature_s = feature_s[:feature_t.shape[0]]
                M_prime = torch.cat((feature_s[1:], feature_s[0].unsqueeze(0)), dim=0)
                dim_l = deepInfoMaxLoss(feature_s, feature_t, M_prime)
            loss = args.err_w * err + args.dim_w * dim_l

            loss.backward()
            optimizer.step()
            deepInfoMaxLossOpt.step()
            sys.stdout.write('\r epoch: %d, [iter: %d / all %d],  err_s_label: %f, err_s_domain: %f, err_t_domain: %f, dim_loss: %f' \
                             % (epoch, i + 1, len_dataloader,err_s_label.data.cpu().numpy(),
                                err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().numpy(), dim_l.data.cpu().numpy()))
            sys.stdout.flush()

        print('\n')
        torch.save(mainModel, args.model_save_path + 'mainModel.pth')
        torch.save(deepInfoMaxLoss, args.model_save_path + 'deepInfoMaxLoss.pth')
        # accu_s = MainTest(dataloader_source, args.model_save_path,'mainModel.pth', device=args.device)
        # print('Accuracy of the %s dataset: %f' % ('source', accu_s))
        accu_t = MainTest(dataloader_target, args.model_save_path, 'mainModel.pth', device=args.device)
        print('Accuracy of the %s dataset: %f' % ('target', accu_t))
        if accu_t > best_accu_t:
            best_accu_t = accu_t
            torch.save(mainModel, args.model_save_path + 'mainModel_best.pth')
            torch.save(deepInfoMaxLoss, args.model_save_path + 'deepInfoMaxLoss_best.pth')