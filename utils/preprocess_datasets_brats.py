
import os
import h5py
import pathlib
from data import transforms
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def main():

    root_dir ='path to /MICCAI_BraTS2020_ValidationData'
    root_out_dir = 'path to /BraTS2020/T1/val'
    sequence = 't1'

    data_dir = pathlib.Path(os.path.join(root_dir))
    out_dir = pathlib.Path(os.path.join(root_out_dir))
    mkdir(out_dir)
    list_shape = []
    list_final_shape = []
    for data_file in tqdm(data_dir.iterdir()):
        fname = os.path.basename(data_file)
        idx = fname
        if not os.path.isfile(os.path.join(out_dir, fname)+'.h5'):
            print('%s processing' % os.path.join(out_dir, fname))
        else:
            continue
        try:
            img = nib.load(os.path.join(data_dir, fname, fname+ '_' + sequence + '.nii.gz'))
        except:
            print('No file: ', os.path.join(data_dir, fname, fname+ '_' + sequence + '.nii.gz'))
            continue
        img_np = np.array(img.dataobj)
        list_shape.append(img_np.shape)
        pad_0 = (0, 0)
        pad_1 = (0, 0)
        # make img to 256 x 256, you may ignore or change this step in your need
        if img_np.shape[0] < 256:
            pad_n = int((256 - img_np.shape[0]) / 2)
            pad_0 = (pad_n, 256 - pad_n - img_np.shape[0])
        if img_np.shape[1] < 256:
            pad_n = int((256 - img_np.shape[1]) / 2)
            pad_1 = (pad_n, 256 - pad_n - img_np.shape[1])
        img_np = np.pad(img_np, (pad_0, pad_1, (0, 0)))
        img_np = np.transpose(img_np, (2, 0, 1)).astype(np.float32)
        list_final_shape.append(img_np.shape)
        img_tensor = transforms.to_tensor(img_np).unsqueeze(-1)
        fake_imag = torch.zeros_like(img_tensor) # make imaginary channel as zero
        img_tensor_complex = torch.cat((img_tensor, fake_imag), dim=-1) # construct complex image
        kspace = transforms.fft2(img_tensor_complex)
        kspace = transforms.to_numpy(kspace)
        max = np.max(img_np)
        norm = np.linalg.norm(img_np)
        acquisition = 'str'
        # construct fastMRI style data format
        with h5py.File(os.path.join(out_dir, idx + '.h5')) as data:
            data.create_dataset('reconstruction_esc', data=img_np)
            data.create_dataset('kspace', data=kspace)

            data.attrs['max'] = max
            data.attrs['norm'] = norm
            data.attrs['acquisition'] = acquisition

            data.flush()
    print('input shapes',set(list_shape))
    print('final shapes', set(list_final_shape))

if __name__ == '__main__':
    main()