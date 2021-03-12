# FL-MRCM

 <a href="https://arxiv.org/abs/2103.02148"> arXiv preprint </a> | <a href="https://sites.google.com/view/fl-mrcm/home"> Project Page </a>

Pytorch Code for the paper ["Multi-institutional Collaborations for Improving Deep Learning-based Magnetic
Resonance Image Reconstruction Using Federated Learning"](https://arxiv.org/abs/2103.02148), will be presented at CVPR 2021


# Requirements

python=3.6  
pytorch=1.4.0

# Inroduction

Fast and accurate reconstruction of magnetic resonance (MR) images from under-sampled data is important in many clinical applications. In recent years, deep learning-based methods have been shown to produce superior performance on MR image reconstruction. However, these methods require large amounts of data which is difficult to collect and share due to the high cost of acquisition and medical data privacy regulations. In order to overcome this challenge, we propose a federated learning (FL) based solution in which we take advantage of the MR data available at different institutions while preserving patients' privacy. However, the generalizability of models trained with the FL setting can still be suboptimal due to domain shift, which results from the data collected at multiple institutions with different sensors, disease types, and acquisition protocols, etc. With the motivation of circumventing this challenge, we propose a cross-site modeling for MR image reconstruction in which the learned intermediate latent features among different source sites are aligned with the distribution of the latent features at the target site. Extensive experiments are conducted to provide various insights about FL for MR image reconstruction. Experimental results demonstrate that the proposed framework is a promising direction to utilize multi-institutional data without compromising patients' privacy for achieving improved MR image reconstruction. 

## Dataset Preparation

Prepare the dataset in the following structure for easy use of the code.The provided data loaders is ready for this this format and you may change it as your need.

```bash


                   |-- T1 
                   |                       
                   |                |--xxx.h5  
Dataset Folder-----|      |--train--|...
                   |      |         |...
                   |      |
                   |      |         |--xxx.h5 
                   |-- T2-|-- val --|...  
                          |         |...
                          |
                          |         |--xxx.h5
                          |--test --|...
                                    |...
```
An example of preprocessing BraTS dataset can be found at <code> utils/preprocess_datasets_brats.py </code>.

## Links for downloading the public Datasets:

1) fastMRI Dataset - <a href="https://fastmri.med.nyu.edu/"> Link </a>  
2) IXI Dataset - <a href="https://brain-development.org/ixi-dataset/"> Link </a> 
3) BraTS Dataset - <a href="https://www.med.upenn.edu/cbica/brats2020/data.html"> Link </a> 
4) HPKS dataset introduced in the paper will be made public after approval.

# Run

## train FL-MR
```bash 
python main_fl_mr.py --phase train --dataset mri --model unet --epochs 50 --challenge singlecoil --local_bs 16 --num_users 4 --local_ep 2 --train_dataset BFHI --test_dataset H --sequence T1  --accelerations 4 --center-fractions 0.08 --val_sample_rate 1.0 --save_dir 'Dir path for saving checkpoints' --verbose
```
## train FL-MRCM
```bash 
python main_fl_mrcm.py --phase train --dataset mri --model unet --epochs 50 --challenge singlecoil --local_bs 16 --num_users 4 --local_ep 2 --train_dataset BFHI --test_dataset B --sequence T1 --accelerations 4 --center-fractions 0.08 --val_sample_rate 1.0 --save_dir 'Dir path for saving checkpoints' --verbose
```
## monitor the traning process
```bash 
tensorboard --logdir 'Dir path for saving checkpoints'
```
## test
```bash 
python main_test.py --phase test --dataset mri --challenge singlecoil --local_bs 16 --model unet --test_dataset I --sequence T1 --accelerations 4 --center-fractions 0.08 --save_dir 'Dir path for saving result'  --checkpoint 'checkpoint path for testing'  --verbose
```
# Ackonwledgements

We give acknowledgements to [fastMRI](https://github.com/facebookresearch/fastMRI), [Federated Learning](https://github.com/shaoxiongji/federated-learning), and [Federated Adversarial Domain Adaptation](https://openreview.net/forum?id=HJezF3VYPB).


# Citation
```bash
@article{guo2021multi,
  title={Multi-institutional Collaborations for Improving Deep Learning-based Magnetic Resonance Image Reconstruction Using Federated Learning},
  author={Guo, Pengfei and Wang, Puyang and Zhou, Jinyuan and Jiang, Shanshan and Patel, Vishal M},
  journal={arXiv preprint arXiv:2103.02148},
  year={2021}
}
```
