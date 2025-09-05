import compressai
import csv
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sarpy.io.general.nitf as nitf
import sys
import time
import torch
import torch.nn as nn
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)
import torch.nn.functional as F
import torchvision


from compressai.zoo import load_state_dict
from dct_fast import ImageDCT
#from ELICUtilis.datasets import SarIQDataset
from option import args
from ELICUtilis.models.NetworkDCT_v2 import SAREliC
from pathlib import Path
from PIL import Image
from sar_evaluation_metrics import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from torchmetrics.functional import structural_similarity_index_measure as SSIM
# from torchmetrics.functional import peak_signal_noise_ratio as PSNR
# from torchmetrics import MeanSquaredError as mse
# from torchvision import transforms
# from typing import List

block_size = 4
dct = ImageDCT(block_size)

device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")

class SarIQDataset(Dataset):
    def __init__(self, root, args) -> None:
        super().__init__()
        self.block_size = args.patch_size
        self.min_val = args.min_val
        self.max_val = args.max_val
        self.file_list = [root]
        self.dataset_size = len(self.file_list)
        
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        input_image_name = self.file_list[idx]
        sar_data         = nitf.NITFReader(input_image_name)
        sar_image        = sar_data.read_raw()
        with open(input_image_name, "rb") as f:
            sar_header   = f.read(sar_data.nitf_details.img_segment_offsets[0])
            if 0:
                print("\n****************\n")
                print(sar_header)
                print("\n****************\n")
            f.seek(sar_data.nitf_details.des_subheader_offsets[0])
            sar_metadata = f.read()
            
            if 0:
                print("\n****************\n")
                print(sar_metadata)
                print("\n****************\n")
            
        sar_image        = np.clip(sar_image, self.min_val, self.max_val)
        sar_image        = (sar_image - self.min_val) / (self.max_val - self.min_val)
        sar_image        = torch.tensor(sar_image).permute(2, 0, 1)

        return {
            'gt_pol': sar_image,
            'name': input_image_name,
            'sar_header': sar_header,
            'sar_metadata': sar_metadata
        }


def decompress():
    print("\n\nDecompression running on: ", device)
    print("Loading models ...")
    compressai.set_entropy_coder(args.entropy_coder)
    data       = torch.load(args.test_model,  map_location=device)
    state_dict = load_state_dict(data['state_dict'])
    model_cls  = SAREliC(N=args.N, M=args.M, input_channels=args.inputchannels, num_slices=5, groups=[0,16,16,32,64,192]).to(device)
    model_cls.load_state_dict(state_dict)
    model      = model_cls.eval()


    bitstream_list = glob.glob(args.save_encoded + '/*.bin')
    bitstream_path = bitstream_list[0]
    with torch.no_grad():
        print("Loading bitstream...")
        print("Bitstream path:", bitstream_path)
        with open(bitstream_path, 'rb') as file:
            bitstream = pickle.load(file)
            strings = bitstream['sar']
            hyper_latent_shape = bitstream['hyper_latent_shape']
            pad_x = bitstream['pad_x']
            pad_y = bitstream['pad_y']
            sar_header = bitstream['sar_header']
            sar_metadata = bitstream['sar_metadata']
        print("Decompressing...")
        start_time = time.time()
        out_dec = model.decompress(strings, hyper_latent_shape)
        dec_time = time.time() - start_time
        print("Computing metrics...")
        # Recompute PSNR and SSIM by unscaling the image, then finding amplitude image, and compare directly            
        max_val   = 5000
        min_val   = -5000
        amp_max_val = torch.sqrt(torch.tensor(max_val ** 2 + min_val ** 2))
        pred_sar_img = out_dec['x_hat'].detach().cpu()

        # pred_sar_img range [0,1] undo the normalization
        pred_sar_img  =  pred_sar_img * (max_val - min_val) + min_val
        pred_sar_img  = pred_sar_img[:,:,:-pad_y,:-pad_x]
        pred_amp = torch.sqrt(pred_sar_img[0,0,:,:]**2 + pred_sar_img[0,1,:,:]**2) 
        predicted_phase = torch.atan2(pred_sar_img[0,1,:,:], pred_sar_img[0,0,:,:])

        f_size_sarelic = Path(bitstream_path)
        bpp = f_size_sarelic.stat().st_size * 8.0 / (pred_sar_img.size(0) * pred_sar_img.size(1) * pred_sar_img.size(2) * pred_sar_img.size(3))

        # load dataset
        test_dataset = SarIQDataset(args.test_image, args)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Compare with the GT IQ data, make sure to undo the normalization
        data         = next(iter(test_loader))
        gt_sar       = data['gt_pol']#.to(device)
        gt_sar       = gt_sar * (max_val - min_val) + min_val
        GT_amp_img   = torch.sqrt(gt_sar[0,0,:,:]**2 + gt_sar[0,1,:,:]**2)
        GT_phase     = torch.atan2(gt_sar[0,1,:,:], gt_sar[0,0,:,:])

        rmse, psnr_val, msssim, sqnr, relative_error = amplitude_error((GT_amp_img/amp_max_val).unsqueeze(0).unsqueeze(0).cuda(), (pred_amp/amp_max_val).unsqueeze(0).unsqueeze(0).cuda(), 5)
        mape         = phase_error(predicted_phase, GT_phase)
        print("Bitrate: %.4f bpp (%d Bytes)\nDecode Time: %.4f\nPSNR: %.4f\nMSSSIM: %.4f\nSQNR: %.4f\nRelative Error: %.4f\nMAPE: %.4f"%(bpp, f_size_sarelic.stat().st_size, dec_time, psnr_val, msssim, sqnr, relative_error, mape))                

if __name__ == '__main__':
    decompress()
    print("Done...")
