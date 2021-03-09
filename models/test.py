#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
from utils import evaluate
import h5py
from tqdm import tqdm


def test_save_result(net_g, datatest, args):
    net_g.eval()
    # testing
    data_loader = DataLoader(datatest, batch_size=args.bs, pin_memory=True)
    test_logs =[]
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            input, target, mean, std,_, fname, slice = batch
            if args.model == 'unet_ad_da':
                output, _ = net_g(input.cuda())
            else:
                output = net_g(input.cuda())
            # sum up batch loss
            test_loss = F.l1_loss(output, target.cuda())
            mean = mean.unsqueeze(1).unsqueeze(2).cuda()
            std = std.unsqueeze(1).unsqueeze(2).cuda()
            test_logs.append({
                'fname': fname,
                'slice': slice,
                'output': (output * std + mean).cpu().detach().numpy(),
                'target': (target.cuda() * std + mean).cpu().numpy(),
                'input': (input.cuda() * std + mean).cpu().numpy(),
                'loss': test_loss.cpu().detach().numpy(),
            })
        losses = []
        outputs = defaultdict(list)
        targets = defaultdict(list)
        inputs = defaultdict(list)
        for log in test_logs:
            losses.append(log['loss'])
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                outputs[fname].append((slice, log['output'][i]))
                targets[fname].append((slice, log['target'][i]))
                inputs[fname].append((slice, log['input'][i]))
        print('loss len: ',len(losses))
        metrics = dict(val_loss=losses, nmse=[], ssim=[], psnr=[])
        outputs_save = defaultdict(list)
        targets_save = defaultdict(list)
        for fname in outputs:
            outputs_save[fname] = np.stack([out for _, out in sorted(outputs[fname])])
            targets_save[fname] = np.stack([tgt for _, tgt in sorted(targets[fname])])
            inputs[fname] = np.stack([tgt for _, tgt in sorted(inputs[fname])])
        for fname in outputs:
            output = np.stack([out for _, out in sorted(outputs[fname])])
            target = np.stack([tgt for _, tgt in sorted(targets[fname])])
            metrics['nmse'].append(evaluate.nmse(target, output))
            metrics['ssim'].append(evaluate.ssim(target, output))
            metrics['psnr'].append(evaluate.psnr(target, output))
        metrics = {metric: np.mean(values) for metric, values in metrics.items()}
        print(metrics, '\n')
        save_reconstructions(outputs_save, args.save_dir /'reconstructions')
        save_reconstructions(targets_save, args.save_dir /'gt')
        save_reconstructions(inputs, args.save_dir /'input')
#
def test_save_vector(net_g, datatest, args):
    net_g.eval()
    # testing
    data_loader = DataLoader(datatest, batch_size=args.local_bs, pin_memory=True)
    test_logs =[]
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(data_loader)):
            input, target, mean, std, _, fname, slice = batch
            output, vector = net_g(input.cuda())

            test_logs.append({
                'fname': fname,
                'slice': slice,
                'vector': vector.cpu().detach().numpy()
            })

        vectors = defaultdict(list)
        for log in test_logs:
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                vectors[fname].append((slice, log['vector'][i]))
        for fname in vectors:
            vectors[fname] = np.stack([tgt for _, tgt in sorted(vectors[fname])])
        print('save the result ...', '\n')
        save_reconstructions(vectors, args.save_dir /'vector')


def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)

class evaluator(object):
    def __init__(self, datatest, args, writer,device):
        self.args = args
        self.device = device
        self.writer = writer
        self.data_loader = DataLoader(datatest, batch_size=args.bs, pin_memory=False)

    def evaluate_recon(self,net_g, epoch=None):
        net_g.eval()
        # testing
        test_logs = []
        with torch.no_grad():
            for idx, batch in enumerate(self.data_loader):
                input, target, mean, std,_, fname, slice = batch
                output,_ = net_g(input.to(self.device))
                # sum up batch loss
                test_loss = F.l1_loss(output, target.to(self.device))
                mean = mean.unsqueeze(1).unsqueeze(2).to(self.device)
                std = std.unsqueeze(1).unsqueeze(2).to(self.device)
                test_logs.append({
                    'fname': fname,
                    'slice': slice,
                    'output': (output * std + mean).cpu().detach().numpy(),
                    'target': (target.to(self.device) * std + mean).cpu().numpy(),
                    'loss': test_loss.cpu().detach().numpy(),
                })
            losses = []
            outputs = defaultdict(list)
            targets = defaultdict(list)
            for log in test_logs:
                losses.append(log['loss'])
                for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                    outputs[fname].append((slice, log['output'][i]))
                    targets[fname].append((slice, log['target'][i]))
            print('loss len: ', len(losses))
            metrics = dict(val_loss=losses, nmse=[], ssim=[], psnr=[])
            for fname in outputs:
                output = np.stack([out for _, out in sorted(outputs[fname])])
                target = np.stack([tgt for _, tgt in sorted(targets[fname])])
                metrics['nmse'].append(evaluate.nmse(target, output))
                metrics['ssim'].append(evaluate.ssim(target, output))
                metrics['psnr'].append(evaluate.psnr(target, output))
            metrics = {metric: np.mean(values) for metric, values in metrics.items()}
            print(metrics, '\n')
            print('No. Slices: ', len(outputs))
            if self.writer != None:
                self.writer.add_scalar('Dev_Loss/NMSE', metrics['nmse'], epoch)
                self.writer.add_scalar('Dev_Loss/SSIM', metrics['ssim'], epoch)
                self.writer.add_scalar('Dev_Loss/PSNR', metrics['psnr'], epoch)
            torch.cuda.empty_cache()

class evaluator_normal(object):
    def __init__(self, datatest, args, writer,device):
        self.args = args
        self.device = device
        self.writer = writer
        self.data_loader = DataLoader(datatest, batch_size=args.bs, pin_memory=False)

    def evaluate_recon(self,net_g, epoch=None):
        net_g.eval()
        # testing
        test_logs = []
        for idx, batch in enumerate(self.data_loader):
            input, target, mean, std,_, fname, slice = batch
            output = net_g(input.to(self.device))
            # sum up batch loss
            test_loss = F.l1_loss(output, target.to(self.device))
            mean = mean.unsqueeze(1).unsqueeze(2).to(self.device)
            std = std.unsqueeze(1).unsqueeze(2).to(self.device)
            test_logs.append({
                'fname': fname,
                'slice': slice,
                'output': (output * std + mean).cpu().detach().numpy(),
                'target': (target.to(self.device) * std + mean).cpu().numpy(),
                'loss': test_loss.cpu().detach().numpy(),
            })
        losses = []
        outputs = defaultdict(list)
        targets = defaultdict(list)
        for log in test_logs:
            losses.append(log['loss'])
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                outputs[fname].append((slice, log['output'][i]))
                targets[fname].append((slice, log['target'][i]))
        print('loss len: ', len(losses))
        metrics = dict(val_loss=losses, nmse=[], ssim=[], psnr=[])
        for fname in outputs:
            output = np.stack([out for _, out in sorted(outputs[fname])])
            target = np.stack([tgt for _, tgt in sorted(targets[fname])])
            metrics['nmse'].append(evaluate.nmse(target, output))
            metrics['ssim'].append(evaluate.ssim(target, output))
            metrics['psnr'].append(evaluate.psnr(target, output))
        metrics = {metric: np.mean(values) for metric, values in metrics.items()}
        print(metrics, '\n')
        print('No. Slices: ', len(outputs))
        if self.writer != None:
            self.writer.add_scalar('Dev_Loss/NMSE', metrics['nmse'], epoch)
            self.writer.add_scalar('Dev_Loss/SSIM', metrics['ssim'], epoch)
            self.writer.add_scalar('Dev_Loss/PSNR', metrics['psnr'], epoch)
        torch.cuda.empty_cache()