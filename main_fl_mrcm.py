#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import os
from utils.options import args_parser
from models.recon_Update import LocalUpdate_ad_da
from models.Fed import FedAvg
from models.test import evaluator
from data.mri_data import SliceData, DataTransform
from data.subsample import create_mask_for_mask_type
from models.unet_model import UnetModel_ad_da, Feature_discriminator
from tensorboardX import SummaryWriter
from pympler import muppy, summary
import pathlib

if __name__ == '__main__':
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    # parse args
    args = args_parser()
    path_dict = {'B': pathlib.Path('Dataset dir B'),
                 'F': pathlib.Path('Dataset dir F'),
                 'H': pathlib.Path('Dataset dir H'),
                 'I': pathlib.Path('Dataset dir I')}
    rate_dict = {'B': 1.0, 'F': 1.0, 'H': 1.0, 'I': 1.0} # control the sample rate for each dataset
    print(rate_dict)
    args.device = torch.device('cuda:{}'.format(args.gpu[0]) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    writer = SummaryWriter(log_dir=args.save_dir/ 'summary')

    def save_networks(net, epoch, local=False, local_no = None):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if local:
            save_filename = '%s_net_D_%s.pth' % (epoch,local_no)
        else:
            save_filename = '%s_net.pth' % (epoch)
        save_path = os.path.join(args.save_dir, save_filename)
        if len(args.gpu) > 1 and torch.cuda.is_available():
            torch.save(net.module.cpu().state_dict(), save_path)
            net.to(args.device)
        else:
            torch.save(net.cpu().state_dict(), save_path)
            net.to(args.device)

    # data loader
    def _create_dataset(data_path,data_transform, data_partition, sequence, sample_rate=None, seed=42):
        dataset = SliceData(
            root=data_path / data_partition,
            transform=data_transform,
            sample_rate=sample_rate,
            challenge=args.challenge,
            sequence=sequence,
            seed=seed
        )
        return dataset

    all_objects = muppy.get_objects()
    sum = summary.summarize(all_objects)

    # load dataset and split users
    if args.dataset == 'mri':
        mask = create_mask_for_mask_type(args.mask_type, args.center_fractions,
                                         args.accelerations)
        train_data_transform = DataTransform(args.resolution, args.challenge, mask, use_seed=False)
        val_data_transform = DataTransform(args.resolution, args.challenge, mask)
        datasets_list = []

        if args.phase == 'train':
            for data in args.train_datasets:
                dataset_train = _create_dataset(path_dict[data]/args.sequence,train_data_transform, 'train', args.sequence,rate_dict[data], args.seed)
                datasets_list.append(dataset_train)
            dataset_val = _create_dataset(path_dict[args.test_dataset]/args.sequence,val_data_transform, 'val', args.sequence, args.val_sample_rate)
    else:
        exit('Error: unrecognized dataset')
    #make target domain dataset has the same number of sample as max train dataset
    target_datasets_list=[]
    for dataset in datasets_list:
        tmp_list = []
        dataset_target_domain = _create_dataset(path_dict[args.test_dataset] / args.sequence, train_data_transform,'train', args.sequence,rate_dict[args.test_dataset])
        while len(tmp_list)<len(dataset.examples):
            for sample in dataset_target_domain.examples:
                tmp_list.append(sample)
                if len(tmp_list) ==len(dataset.examples):
                    break
        dataset_target_domain.examples = tmp_list
        target_datasets_list.append(dataset_target_domain)

    assert (len(datasets_list)==args.num_users)
    # build model
    if args.model == 'unet':
        net_glob = UnetModel_ad_da(
            in_chans=1,
            out_chans=1,
            chans=32,
            num_pool_layers=4,
            drop_prob=0.0
        ).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()
    G_s = []
    FD = []
    for i in range(args.num_users):
        if len(args.gpu) > 1:
            G_s.append(torch.nn.DataParallel(UnetModel_ad_da(in_chans=1, out_chans=1, chans=32, num_pool_layers=4,
                                                       drop_prob=0.0).to(args.device),args.gpu))
            FD.append(torch.nn.DataParallel(Feature_discriminator().to(args.device),args.gpu))
        else:
            G_s.append(UnetModel_ad_da(in_chans=1,out_chans=1,chans=32,num_pool_layers=4,drop_prob=0.0).to(args.device))
            FD.append(Feature_discriminator().to(args.device))
    # setting optimizer
    opt_g_s = []
    opt_FD= []
    for i in range(args.num_users):
        opt_g_s.append(torch.optim.RMSprop(G_s[i].parameters(), lr=args.lr))
        opt_FD.append(torch.optim.RMSprop(FD[i].parameters(), lr=args.lr*10))
    # copy weights
    if len(args.gpu) > 1:
        net_glob = torch.nn.DataParallel(net_glob, args.gpu)
        w_glob = net_glob.state_dict()
    else:
        w_glob = net_glob.state_dict()

    # initilize parameters
    for G in G_s:
        for net, net_cardinal in zip(G.named_parameters(), net_glob.named_parameters()):
            net[1].data = net_cardinal[1].data.clone()
    # training
    if args.phase == 'train':
        start_epoch = -1
        if args.continues:
            if len(args.gpu) > 1:
                net_glob.module.load_state_dict(torch.load(args.checkpoint))
                print('Load checkpoint :', args.checkpoint)
                for i, net_d in enumerate(FD):
                    path = args.checkpoint.split('.')[0]+'_D_%s.pth'%(i)
                    print('Load checkpoint :', path)
                    net_d.module.load_state_dict(torch.load(path))
                start_epoch = int(args.checkpoint.split('/')[-1].split('_')[0])
            else:
                net_glob.load_state_dict(torch.load(args.checkpoint))
                print('Load checkpoint :', args.checkpoint)
                for i, net_d in enumerate(FD):
                    path = args.checkpoint.split('.')[0] + '_D_%s.pth' % (i)
                    print('Load checkpoint :', path)
                    net_d.load_state_dict(torch.load(path))
                start_epoch = int(args.checkpoint.split('/')[-1].split('_')[0])

        for iter in range(start_epoch+1,args.epochs):
            w_locals, loss_locals = [], []
            for idx, dataset_train in enumerate(datasets_list):
                flag = args.train_datasets[idx] == args.test_dataset # for disable adv loss for target dataset
                local = LocalUpdate_ad_da(args=args, device=args.device, dataset=dataset_train,
                                       dataset_target = target_datasets_list[idx], optimizer=opt_g_s[idx],optimizer_fd=opt_FD[idx],flag=flag)
                # models communication
                G_s[idx].load_state_dict(net_glob.state_dict())
                # global update
                w, loss = local.train(net=G_s[idx],net_fd=FD[idx] ,epoch=iter, idx=idx, writer=writer)
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            # update global weights
            w_glob = FedAvg(w_locals)
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)
            # track memory leaking
            all_objects = muppy.get_objects()
            sum = summary.summarize(all_objects)[:10]
            for item in sum:
                writer.add_scalar('Memory/'+ item[0], item[2], iter)
            # print loss
            loss_avg = np.sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            print('saving the model at the end of epoch %d' % (iter))
            save_networks(net_glob, iter)
            for i, net_d in enumerate(FD):
                save_networks(net_d, iter, local=True, local_no=i)
            print('Evaluation ...')
            validation = evaluator(dataset_val, args, writer,args.device)
            validation.evaluate_recon(net_glob,iter)
            # empty gpu cache
            del w_locals, loss_locals, w_glob, local, validation
            torch.cuda.empty_cache()
        writer.close()


