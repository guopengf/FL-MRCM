#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F


class LocalUpdate(object):
    def __init__(self, args, device, dataset=None):
        self.args = args
        self.device = device
        self.loss_func = nn.L1Loss().to(device)
        self.selected_clients = []
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True, pin_memory=True)

    def train(self, net, epoch, idx, writer):
        net.train()
        # train and update
        optimizer = torch.optim.RMSprop(net.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            iter_data_time = time.time()
            for batch_idx, batch in enumerate(self.ldr_train):
                input, target, mean, std, norm, fname, slice = batch
                net.zero_grad()
                output = net(input.to(self.device))
                loss = self.loss_func(output, target.to(self.device))
                loss.backward()
                optimizer.step()
                scheduler.step(epoch)
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} Local: {} idx: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, iter, idx, batch_idx * len(input), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.detach().item()))
                    t_comp = (time.time() - iter_data_time)
                    iter_data_time = time.time()
                    print('itr time: ',t_comp)
                    print('lr: ',optimizer.param_groups[0]['lr'])
                batch_loss.append(loss.detach().item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        writer.add_scalar('TrainLoss/L1/'+ self.args.train_datasets[idx], sum(epoch_loss) / len(epoch_loss), epoch)
        torch.cuda.empty_cache()

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), net

class LocalUpdate_ad_da(object):
    # v1
    def __init__(self, args, device, dataset, dataset_target,optimizer,optimizer_fd,flag):
        self.args = args
        self.device = device
        self.loss_func = nn.L1Loss().to(device)
        self.adv_loss = nn.BCEWithLogitsLoss().to(device)
        self.selected_clients = []
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True, pin_memory=True)
        self.target_domain = DataLoader(dataset_target, batch_size=self.args.local_bs, shuffle=True, pin_memory=True)
        self.optimizer = optimizer
        self.optimizer_fd = optimizer_fd
        self.flag = flag

    def nmse_loss(self,output,target,std, mean, norm):
        return F.mse_loss((output * std + mean) / norm, (target * std + mean) / norm)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train(self, net, net_fd, epoch, idx, writer):
        net.train()
        net_fd.train()
        # train and update
        optimizer = self.optimizer
        optimizer_fd = self.optimizer_fd

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

        epoch_loss = []
        epoch_loss_L1 = []
        epoch_loss_adv_g = []
        epoch_loss_adv_d = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_loss_L1 = []
            batch_loss_adv_g = []
            batch_loss_adv_d = []
            iter_data_time = time.time()
            for batch_idx, (batch, batch_t) in enumerate(zip(self.ldr_train,self.target_domain)):
                input, target, mean, std, norm, fname, slice = batch
                input_t,target_t, _,_,_,_,_ = batch_t
                net.zero_grad()
                net_fd.zero_grad()
                output,z = net(input.to(self.device))
                output_t, z_t = net(input_t.to(self.device))
                mean = mean.unsqueeze(1).unsqueeze(2).to(self.device)
                std = std.unsqueeze(1).unsqueeze(2).to(self.device)
                norm = norm.unsqueeze(1).unsqueeze(2).to(self.device)

                # prepare labels for fd
                src_domain_code = np.repeat(np.array([[*([1]), *([0])]]), input.shape[0], axis=0)
                tgt_domain_code = np.repeat(np.array([[*([0]), *([1])]]), input_t.shape[0], axis=0)
                src_domain_code = Variable(torch.FloatTensor(src_domain_code).to(self.device), requires_grad=False)
                tgt_domain_code = Variable(torch.FloatTensor(tgt_domain_code).to(self.device), requires_grad=False)

                # update DF for source site
                df_loss = 0
                if not self.flag:
                    self.set_requires_grad(net_fd, True)
                    src_domain_pred = net_fd(z.detach())
                    tgt_domain_pred = net_fd(z_t.detach())
                    # adv loss for DF
                    df_loss_src = self.adv_loss(src_domain_pred, 1 - src_domain_code)
                    df_loss_tgt = self.adv_loss(tgt_domain_pred, 1 - tgt_domain_code)
                    df_loss = ((df_loss_src + df_loss_tgt)/2)

                    df_loss.backward()
                    optimizer_fd.step()
                batch_loss_adv_d.append(df_loss)

                # update Recon
                self.set_requires_grad(net_fd, False)
                loss_adv_g = 0
                if not self.flag:
                    # adv for recon net
                    src_domain_pred = net_fd(z)
                    tgt_domain_pred = net_fd(z_t)
                    df_loss_src = self.adv_loss(src_domain_pred, src_domain_code)
                    df_loss_tgt = self.adv_loss(tgt_domain_pred, tgt_domain_code)
                    loss_adv_g =((df_loss_src+df_loss_tgt)/2)
                batch_loss_adv_g.append(loss_adv_g)

                L1_loss = self.loss_func(output, target.to(self.device))
                batch_loss_L1.append(L1_loss.detach().item())
                if not self.flag:
                    loss = L1_loss + loss_adv_g
                else:
                    loss = L1_loss
                batch_loss.append(loss.detach().item())
                loss.backward()
                optimizer.step()
                scheduler.step(epoch)
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} Local: {} idx: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, iter, idx, batch_idx * len(input), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.detach().item()))
                    t_comp = (time.time() - iter_data_time)
                    iter_data_time = time.time()
                    print('itr time: ',t_comp)
                    print('lr: ',optimizer.param_groups[0]['lr'])

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_loss_L1.append(sum(batch_loss_L1)/len(batch_loss_L1))
            epoch_loss_adv_g.append(sum(batch_loss_adv_g)/len(batch_loss_adv_g))
            epoch_loss_adv_d.append(sum(batch_loss_adv_d)/len(batch_loss_adv_d))
        writer.add_scalar('TrainLoss/L1/'+ self.args.train_datasets[idx], sum(epoch_loss_L1) / len(epoch_loss_L1), epoch)
        writer.add_scalar('TrainLoss/total_loss/' + self.args.train_datasets[idx], sum(epoch_loss) / len(epoch_loss),epoch)
        writer.add_scalar('TrainLoss/adv_g/' + self.args.train_datasets[idx], sum(epoch_loss_adv_g) / len(epoch_loss_adv_g), epoch)
        writer.add_scalar('TrainLoss/adv_d/' + self.args.train_datasets[idx], sum(epoch_loss_adv_d) / len(epoch_loss_adv_d), epoch)
        torch.cuda.empty_cache()

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

