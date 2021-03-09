#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import os
from utils.options import args_parser
from models.test import test_save_result, test_save_vector
from data.mri_data import SliceData, DataTransform
from data.subsample import create_mask_for_mask_type
from models.unet_model import UnetModel, UnetModel_ad_da
import pathlib

if __name__ == '__main__':
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    # parse args
    args = args_parser()
    path_dict = {'B': pathlib.Path('Dataset dir B'),
                 'F': pathlib.Path('Dataset dir F'),
                 'H': pathlib.Path('Dataset dir H'),
                 'I': pathlib.Path('Dataset dir I')}
    args.device = torch.device('cuda:{}'.format(args.gpu[0]) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # data loader
    def _create_dataset(data_path,data_transform, data_partition, sequence, sample_rate=None):
        sample_rate = sample_rate or args.sample_rate
        dataset = SliceData(
            root=data_path / data_partition,
            transform=data_transform,
            sample_rate=sample_rate,
            challenge=args.challenge,
            sequence=sequence
        )
        return dataset

    # load dataset and split users
    if args.dataset == 'mri':
        mask = create_mask_for_mask_type(args.mask_type, args.center_fractions,
                                         args.accelerations)
        val_data_transform = DataTransform(args.resolution, args.challenge, mask)

        dataset_val = _create_dataset(path_dict[args.test_dataset]/args.sequence,val_data_transform, 'test', args.sequence,1.0)
    else:
        exit('Error: unrecognized dataset')

    if args.model == 'unet': # for fl_mr
        net_glob = UnetModel(
            in_chans=1,
            out_chans=1,
            chans=32,
            num_pool_layers=4,
            drop_prob=0.0
        ).to(args.device)
    elif args.model == 'unet_ad_da': # for fl_mrcm
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

    # copy weights
    if len(args.gpu) > 1:
        net_glob = torch.nn.DataParallel(net_glob, args.gpu)
    if args.phase == 'test':
        # testing
        net_glob.eval()
        if len(args.gpu) > 1:
            net_glob.module.load_state_dict(torch.load(args.checkpoint))
        else:
            net_glob.load_state_dict(torch.load(args.checkpoint))
        print('Load checkpoint for test:', args.checkpoint)
        test_save_result(net_glob, dataset_val, args)



