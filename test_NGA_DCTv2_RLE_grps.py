"""
Perform inference by writing and loading file from disk. This file is to be used for only the SAR_amp dataset for now due to normalization and denormalization of the data.
"""

from ELICUtilis.encoding.rle import build_huffman_codebook
import compressai
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import time
import torch
import torch.nn as nn
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
# from bitarray import bitarray
import h5py

from compressai.zoo import load_state_dict
from dct_fast import ImageDCT
from ELICUtilis.datasets import SarIQDataset
from option_NGA_DCTv2 import args
from ELICUtilis.models.NetworkDCT_v2_RLE_grps import SAREliC
from ELICUtilis.encoding.rle import *
from pathlib import Path
from PIL import Image
from sar_evaluation_metrics import *
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
from torchmetrics import MeanSquaredError as mse
from torchvision import transforms
from typing import List

block_size = 4
dct = ImageDCT(block_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    compressai.set_entropy_coder(args.entropy_coder)
    data       = torch.load(args.test_model)
    state_dict = load_state_dict(data['state_dict'])
    model_cls  = SAREliC(N=args.N, M=args.M, input_channels=args.inputchannels, num_slices=5, groups=[0,16,16,32,64,192]).to(device)
    model_cls.load_state_dict(state_dict)
    model      = model_cls.eval()

    # load dataset
    test_dataset = SarIQDataset(args.test_dataset, train=False, names=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.test_batch_size,shuffle=False)

    # create output directory
    if args.save_encoded is not None:
        os.makedirs(args.save_encoded, exist_ok=True)
    
    model_name = args.test_model.split('/')[-1]
    lmbda = int(''.join(filter(str.isdigit, model_name)))
    csv_name = f"{args.primary_pol}_bypass{args.bypass_grps}_lambda{lmbda}.csv"
    csv_path = os.path.join("/scratch/zb7df/data/results/", csv_name)

    if "0" in args.bypass_grps:
        bypass_grps = []
    else:
        bypass_grps = [int(x)-1 for x in args.bypass_grps]
        assert all(x in [0,1,2,3,4] for x in bypass_grps), "Valid bypass_grps: [0,1,2,3,4,5]"

    ## Build Global Codebooks ###
    print("### BUILDING CODEBOOKS ###")
    y = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            im_name      = data['name']
            data         = data['gt_pol'].to(device)
            if args.primary_pol == "HH":
                gt_sar   = data[:,0:2,:,:]
            elif args.primary_pol == "HV":
                gt_sar   = data[:,2:4,:,:]
            elif args.primary_pol == "VH":
                gt_sar   = data[:,4:6,:,:]
            elif args.primary_pol == "VV":
                gt_sar   = data[:,6:8,:,:]

            gt_sar = gt_sar.to(device)
            image_dct = dct.dct_2d(gt_sar)
            y_i = model.g_a(image_dct)
            y.append(y_i)

        y = torch.cat(y, dim=0)
        y_hat = torch.flatten(y).round().int().cpu().numpy()

        vals, lens = rle_encode(y_hat)
        vals_min = np.min(vals)
        vals_adj = vals - vals_min
        vals_codebook = build_huffman_codebook(vals_adj)
        lens_codebook = build_huffman_codebook(lens)
        
    print("### CODEBOOKS COMPLETE ###\n")
    #############################

    y = []
    y_hat = []
    z_hat = []
    with open(csv_path, 'w',) as file:
        writer = csv.writer(file)
        writer.writerow(["Encoded_file", "bpp", "psnr", 'ssim', 'sqnr','mape_phase', 'enc_time', 'dec_time',
                         "enc1_m", "enc1_s", "enc2_m", "enc2_s", "enc3_m", "enc3_s", "enc4_m", "enc4_s", "enc5_m", "enc5_s",
                         "dec1_m", "dec1_s", "dec2_m", "dec2_s", "dec3_m", "dec3_s", "dec4_m", "dec4_s", "dec5_m", "dec5_s"])

        with torch.no_grad():
            for i, data in tqdm(enumerate(test_loader)):
                # if i >= 100:
                #     break
                im_name      = data['name']
                data         = data['gt_pol'].to(device)
                if args.primary_pol == "HH":
                    gt_sar   = data[:,0:2,:,:]
                elif args.primary_pol == "HV":
                    gt_sar   = data[:,2:4,:,:]
                elif args.primary_pol == "VH":
                    gt_sar   = data[:,4:6,:,:]
                elif args.primary_pol == "VV":
                    gt_sar   = data[:,6:8,:,:]

                gt_sar = gt_sar.to(device)
                image_dct = dct.dct_2d(gt_sar)
                # avg_time = 0
                # for i in range(10):
                #     #start = time.time()
                #     out_enc = model.compress(image_dct)
                #     #avg_time += time.time() - start
                #     avg_time += out_enc['time']['y_enc'] + out_enc['time']['z_enc'] + out_enc['time']['z_dec'] + out_enc['time']['params']
                # print("Avg encode time: %.4f"%(avg_time/10))
                
                start_time = time.time()
                out_enc = model.compress(image_dct, bypass_grps, vals_codebook, lens_codebook, vals_min)
                enc_time = time.time() - start_time

                # save the encoded files
                if args.save_encoded is not None:
                    start_time = time.time()
                    encode_path = os.path.join(args.save_encoded, os.path.splitext(im_name[0])[0] + '.bin')
                    with open(encode_path, 'wb') as file:
                        pickle.dump(out_enc['strings'], file)

                    save_time = time.time() - start_time
                    # load them back and decompress
                    with open(encode_path, 'rb') as file:
                        strings = pickle.load(file)

                else:
                    strings = out_enc['strings']
                    save_time = 0


                start_time = time.time()
                out_dec = model.decompress(strings, torch.Size([args.patch_size[0]//64, args.patch_size[0]//64]),
                                                            bypass_grps=bypass_grps, out_enc=out_enc, vals_codebook=vals_codebook, 
                                                            lens_codebook=lens_codebook, vals_min=vals_min)
                dec_time = time.time() - start_time

                y.append(out_enc["y"])
                y_hat.append(out_enc["y_hat"])
                z_hat.append(out_enc["z_hat"])

                f_size_sarelic = Path(encode_path)
                bpp = f_size_sarelic.stat().st_size * 8.0 / (gt_sar.size(0) * gt_sar.size(1) * gt_sar.size(2) * gt_sar.size(3))
                # print("Bitrate: %.4f bpp per band\n"%(bpp))

                # Recompute PSNR and SSIM by unscaling the image, then finding amplitude image, and compare directly            
                max_val   = 5000
                min_val   = -5000
                amp_max_val = torch.sqrt(torch.tensor(max_val ** 2 + min_val ** 2))
                pred_sar_img = out_dec['x_hat'].detach().cpu()

                # pred_sar_img range [0,1] undo the normalization
                pred_sar_img  =  pred_sar_img * (max_val - min_val) + min_val

                pred_amp = torch.sqrt(pred_sar_img[0,0,:,:]**2 + pred_sar_img[0,1,:,:]**2) # now amplitude image

                ## Compare with the GT IQ data, make sure to undo the normalization
                gt_sar = gt_sar.detach().cpu()
                gt_sar = gt_sar * (max_val - min_val) + min_val
                GT_amp_img = torch.sqrt(gt_sar[0,0,:,:]**2 + gt_sar[0,1,:,:]**2)

                # This is unnormalized direct comparison of amplitude images from IQ GT and IQ predicted
                #np.save('./gimp/sarelic_amp_test_best.npy', (pred_amp/amp_max_val).numpy())
                rmse, psnr_val, msssim, sqnr, relative_error = amplitude_error((GT_amp_img/amp_max_val).unsqueeze(0).unsqueeze(0).cuda(), (pred_amp/amp_max_val).unsqueeze(0).unsqueeze(0).cuda(), 5)
                # amplitude_error(((GT_amp_img-torch.min(GT_amp_img))/(torch.max(GT_amp_img)-torch.min(GT_amp_img))).unsqueeze(0).unsqueeze(0).cuda(), 
                #                                                             ((pred_amp-torch.min(pred_amp))/(torch.max(pred_amp)-torch.min(pred_amp))).unsqueeze(0).unsqueeze(0).cuda(), 
                #                                                             5)

                # Here find and compare phase information
                predicted_phase = torch.atan2(pred_sar_img[0,1,:,:], pred_sar_img[0,0,:,:])
                #np.save('./gimp/sarelic_phase_test_best.npy', predicted_phase.numpy())
                # np.save('sarelic_phase_test.npy', predicted_phase.numpy())
                # np.save('sarelic_amp_test.npy', (pred_amp/amp_max_val).numpy())
                GT_phase = torch.atan2(gt_sar[0,1,:,:], gt_sar[0,0,:,:])

                mape = phase_error(predicted_phase, GT_phase)

                slice_enc_times = out_enc["time"]["slices"]
                enc1_m = slice_enc_times[0]["mean"]
                enc2_m = slice_enc_times[1]["mean"]
                enc3_m = slice_enc_times[2]["mean"]
                enc4_m = slice_enc_times[3]["mean"]
                enc5_m = slice_enc_times[4]["mean"]
                enc1_s = slice_enc_times[0]["std"]
                enc2_s = slice_enc_times[1]["std"]
                enc3_s = slice_enc_times[2]["std"]
                enc4_s = slice_enc_times[3]["std"]
                enc5_s = slice_enc_times[4]["std"]

                slice_dec_times = out_dec["time"]["slices"]
                dec1_m = slice_dec_times[0]["mean"]
                dec2_m = slice_dec_times[1]["mean"]
                dec3_m = slice_dec_times[2]["mean"]
                dec4_m = slice_dec_times[3]["mean"]
                dec5_m = slice_dec_times[4]["mean"]
                dec1_s = slice_dec_times[0]["std"]
                dec2_s = slice_dec_times[1]["std"]
                dec3_s = slice_dec_times[2]["std"]
                dec4_s = slice_dec_times[3]["std"]
                dec5_s = slice_dec_times[4]["std"]

                # print(f"Image: {im_name[0]}\nBPP : {bpp:.4f}\nEncoder time: {enc_time:.4f}\nFile save time: {save_time:.4f}\nDecoder time: {dec_time:.4f}\nPSNR: {psnr_val:.4f}\nRMSE: {rmse:.4f}\nSSIM: {msssim:.4f}\nSQNR: {sqnr:.4f}\nRE  : {relative_error:.4f}\nMAPE: {mape:.4f}, {180*mape/math.pi:.4f}, pi/{math.pi/mape:.4f}")
                writer.writerow([im_name[0], bpp, psnr_val, msssim.item(), sqnr.item(), mape.item(), 
                                 enc_time, dec_time, enc1_m, enc1_s, enc2_m, enc2_s, enc3_m, enc3_s, enc4_m, enc4_s, enc5_m, enc5_s,
                                 dec1_m, dec1_s, dec2_m, dec2_s, dec3_m, dec3_s, dec4_m, dec4_s, dec5_m, dec5_s])

                del data, gt_sar, image_dct, out_enc, out_dec
                torch.cuda.empty_cache()

    output_file = os.path.join(f"/scratch/zb7df/data/latents/latent_tensors_lambda{lmbda}.h5")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset("y", data=y)
        f.create_dataset("y_hat", data=y_hat)
        f.create_dataset("z_hat", data=z_hat)

if __name__ == '__main__':
    overall_start = time.time()
    main()
    print("Done...\n")
    overall_time = time.time() - overall_start
    print(f"Overall time: {overall_time:.4f} sec")
