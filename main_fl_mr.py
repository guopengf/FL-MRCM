#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import torch
import os
from utils.options import args_parser
from models.recon_Update import LocalUpdate
from models.Fed import FedAvg
from models.test import evaluator_normal as evaluator
from data.mri_data import SliceData, DataTransform
from data.subsample import create_mask_for_mask_type
from models.unet_model import UnetModel
from tensorboardX import SummaryWriter
import pathlib

if __name__ == '__main__':
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    # parse args
    args = args_parser()
    path_dict = {'B': pathlib.Path('Dataset dir B'),
                 'F': pathlib.Path('Dataset dir F'),
                 'H': pathlib.Path('Dataset dir H'),
                 'I': pathlib.Path('Dataset dir I')}
    rate_dict = {'B': 1.0,'F': 1.0,'H': 1.0, 'I': 1.0} # control the sample rate for each dataset

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
            save_filename = '%s_C%s_net.pth' % (epoch,local_no)
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
    def _create_dataset(data_path,data_transform, data_partition, sequence, sample_rate=None):
        dataset = SliceData(
            root=data_path / data_partition,
            transform=data_transform,
            sample_rate=sample_rate,
            challenge=args.challenge,
            sequence =sequence
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
                dataset_train = _create_dataset(path_dict[data]/args.sequence,train_data_transform, 'train', args.sequence,rate_dict[data])
                datasets_list.append(dataset_train)
            dataset_val = _create_dataset(path_dict[args.test_dataset]/args.sequence,val_data_transform, 'val', args.sequence, args.val_sample_rate)
    else:
        exit('Error: unrecognized dataset')

    assert (len(datasets_list)==args.num_users)
    # build model
    if args.model == 'unet':
        net_glob = UnetModel(
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

    # copy weights
    if len(args.gpu) > 1:
        net_glob = torch.nn.DataParallel(net_glob, args.gpu)
        w_glob = net_glob.module.state_dict()
    else:
        w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.phase == 'train':
        start_epoch = -1
        if args.continues:
            if len(args.gpu) > 1:
                net_glob.module.load_state_dict(torch.load(args.checkpoint))
            else:
                net_glob.load_state_dict(torch.load(args.checkpoint))
            print('Load checkpoint :', args.checkpoint)
            start_epoch = int(args.checkpoint.split('/')[-1].split('_')[0])

        for iter in range(start_epoch+1,args.epochs):
            w_locals, loss_locals = [], []
            for idx, dataset_train in enumerate(datasets_list):
                local = LocalUpdate(args=args, device=args.device, dataset=dataset_train)
                # global update
                w, loss, _ = local.train(net=copy.deepcopy(net_glob).to(args.device),epoch=iter, idx=idx, writer=writer)
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            # update global weights
            w_glob = FedAvg(w_locals)
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)
            # print loss
            loss_avg = np.sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            print('saving the model at the end of epoch %d' % (iter))
            save_networks(net_glob, iter)
            print('Evaluation ...')
            validation = evaluator(dataset_val, args, writer,args.device)
            validation.evaluate_recon(net_glob,iter)
        writer.close()


