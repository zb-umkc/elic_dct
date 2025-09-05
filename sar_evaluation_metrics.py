import math
import numpy as np
from pytorch_msssim import MS_SSIM
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import torch
import torch.nn as nn

mse_loss = nn.MSELoss()
ms_ssim_amp_module = MS_SSIM(data_range=1.0, size_average=True, channel=1)

def phase_error(phase1, phase2):
    return torch.mean(torch.abs(phase1 - phase2))

def compute_SQNR_vectorized(signal, quantized_signal,neighborhood_size=5):
    # signal = torch.tensor(signal, dtype=torch.float32)
    # quantized_signal = torch.tensor(quantized_signal, dtype=torch.float32)
    #half_size = neighborhood_size // 2

    # Compute local signal power using convolution
    local_signal_power = torch.nn.functional.conv2d(signal**2, 
                                                    torch.ones(1, 1, neighborhood_size, neighborhood_size).cuda())

    # Compute quantization noise power
    noise = signal - quantized_signal
    local_quantization_noise_power = torch.nn.functional.conv2d(noise**2, 
                                                                torch.ones(1, 1, neighborhood_size, neighborhood_size).cuda())
    relative_error = torch.mean(local_quantization_noise_power/local_signal_power)
    sqnr = torch.mean(10*torch.log10(local_signal_power/neighborhood_size**2) - 10*torch.log10(local_quantization_noise_power/neighborhood_size**2))
    #sqnr = torch.mean((local_signal_power) / (local_quantization_noise_power))
    
    return sqnr, relative_error

def amplitude_error(target, pred, neighborhood_size):
    pred = torch.clamp(pred, 0, 1)
    sqnr, relative_error = compute_SQNR_vectorized(target, pred, neighborhood_size)
    mse = mse_loss(target, pred)
    rmse = torch.sqrt(mse)
    psnr = 10*np.log10(1/mse.item())
    msssim = ms_ssim_amp_module(target, pred)
    return rmse, psnr, msssim, sqnr, relative_error

if __name__ == "__main__":
    target = torch.rand(1, 1, 256, 256).cuda()
    pred = torch.rand(1, 1, 256, 256).cuda()
    rmse, psnr, msssim, sqnr, relative_error = amplitude_error(target, pred)
    mape = phase_error(target, pred)
    print("PSNR: %.4f"%(psnr))
    print("RMSE: %.4f"%(rmse))
    print("MS-SSIM: %.4f"%(msssim))
    print("SQNR: %.4f"%(sqnr))
    print("Relative Error: %.4f"%(relative_error))
    print("MAPE: %.4f"%(mape))