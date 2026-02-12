# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


###############################################################################
# Load the gcc version for compressai                                         #
# module load gcc/11.2.0-gcc-8.3.0-cuda-11.2.2-ay27                           #
###############################################################################

import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from ELICUtils.datasets import *
from ELICUtils.encoding import *
from ELICUtils.metrics import *
from ELICUtils.models.DCT_grps import *
# from ELICUtils.models.DCT_RLE import *
from ELICUtils.utils import *
from options import args


block_size = 4
dct = ImageDCT(block_size)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
if args.seed is not None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    
PSNR = PSNR(data_range=1.0)
SSIM = SSIM(data_range=1.0)

class RateDistortionLoss(nn.Module):
    """
    ### Rate Distortion Loss
    Standard RD loss with Lagrangian parameter.
    - larger lambda -> more emphasis on distortion
    - Loss = lambda * MSE + BPP
    
    """

    def __init__(self, lmbda=1e-2, loss=1, phase_loss=False, scheduled_phase=0.0, scaled_amp=False):
        super().__init__()
        self.mse             = nn.MSELoss()    # MSE loss
        self.l1_loss         = nn.L1Loss()     # L1 loss
        self.nmse_loss       = NMSELoss(args)
        self.corr_loss       = CorrLoss(args)
        self.lmbda           = lmbda * (15e-3)
        self.phase_loss      = phase_loss
        self.scheduled_phase = scheduled_phase
        self.scaled_amp      = scaled_amp
        self.loss            = loss

    def generate_mask(self, image, percentiles=torch.tensor([605, 1092, 2092]), weights=torch.tensor([1/16, 1/8, 1/4, 1/2])):
        mask = torch.zeros_like(image, dtype=float)  # Initialize mask with zeros

        # Create boolean masks for each percentile range
        mask_1 = torch.lt(image, percentiles[0])
        mask_2 = torch.logical_and(torch.le(percentiles[0], image), torch.lt(image, percentiles[1]))
        mask_3 = torch.logical_and(torch.le(percentiles[1], image), torch.lt(image,percentiles[2]))
        mask_4 = torch.ge(image, percentiles[2])

        # Assign weights based on the masks
        mask[mask_1] = weights[0]
        mask[mask_2] = weights[1]
        mask[mask_3] = weights[2]
        mask[mask_4] = weights[3]

        return mask

            

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values() # information/ total pixels 
        )
        out["y_bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["y"]
        )
        out["z_bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["z"]
        )
        out["mse_loss"] = self.mse(output["x_hat"], target) * 255 ** 2

        if self.loss == 2:
            out["L1_loss"] = self.l1_loss(output["x_hat"], target) * 255
            out["loss"] = self.lmbda * out["L1_loss"] + out["bpp_loss"]
        elif self.loss == 4:
            # Compute nmse loss
            unnorm_output = output["x_hat"] * (args.max_val - args.min_val) + args.min_val
            unnorm_target = target * (args.max_val - args.min_val) + args.min_val
            out["nmse_loss"] = self.nmse_loss(unnorm_output, unnorm_target)
            out["loss"] = self.lmbda * out["nmse_loss"] + out["bpp_loss"]
        elif self.loss == 5:
            # Compute correlation loss
            unnorm_output = output["x_hat"] * (args.max_val - args.min_val) + args.min_val
            unnorm_target = target * (args.max_val - args.min_val) + args.min_val
            out["corr_loss"] = 1 - self.corr_loss(unnorm_output, unnorm_target)
            out["loss"] = self.lmbda * out["corr_loss"] + out["bpp_loss"]
        else:
            out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """
    ### Separate parameters for the main optimizer and the auxiliary optimizer.
    - Main optimizer: Main parameters of the compression network
    - Auxiliary optimizer: Parameters of the entropy bottleneck for probability distribution estimation

    Return two optimizers
    """

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict  = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate, betas=(0.9, 0.999),
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate, betas=(0.9, 0.999),
    )
    return optimizer, aux_optimizer


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, noisequant='noise'):
    model.train()
    device = next(model.parameters()).device
    PSNR.to(device)
    SSIM.to(device)
    train_loss          = AverageMeter()
    train_bpp_loss      = AverageMeter()
    train_y_bpp_loss    = AverageMeter()
    train_z_bpp_loss    = AverageMeter()
    train_mse_loss      = AverageMeter()
    train_psnr_avg      = AverageMeter()
    train_ssim_avg      = AverageMeter()

    start = time.time()
    for i, data in enumerate(train_dataloader):
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
        d_dct = dct.dct_2d(gt_sar)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out_net = model(d_dct, noisequant)

        out_criterion = criterion(out_net, gt_sar)
        train_bpp_loss.update(out_criterion["bpp_loss"].item())
        train_y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
        train_z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
        train_loss.update(out_criterion["loss"].item())
        train_mse_loss.update(out_criterion["mse_loss"].item())
        train_psnr_avg.update(PSNR(out_net["x_hat"], gt_sar).item())
        train_ssim_avg.update(SSIM(out_net["x_hat"], gt_sar).item())

        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0:
            print(f"Lambda: {args.lmbda} | losstype: {args.loss}| "
                f"Train epoch {epoch}: ["
                f"{i * len(gt_sar):<5}/{len(train_dataloader.dataset):<5}"
                f" ({100. * i / len(train_dataloader):>3.0f}%)]"
                f"Loss: {out_criterion['loss'].item():<7.3f} |"
                f"MSE loss: {out_criterion['mse_loss'].item():<7.3f} |"
                f"Bpp loss: {out_criterion['bpp_loss'].item():<7.3f} |"
                f"y_Bpp loss: {out_criterion['y_bpp_loss'].item():<7.4f} |"
                f"z_Bpp loss: {out_criterion['z_bpp_loss'].item():<7.4f} |"
                f"Aux loss: {aux_loss.item():<5.2f}")
    print(f"Lambda: {args.lmbda} | losstype: {args.loss}| "
        f"Train epoch {epoch}:"
        f"Loss: {train_loss.avg:<7.3f} |"
        f"MSE loss: {train_mse_loss.avg:<7.3f} |"
        f"Bpp loss: {train_bpp_loss.avg:<7.4f} |"
        f"y_Bpp loss: {train_y_bpp_loss.avg:<8.5f} |"
        f"z_Bpp loss: {train_z_bpp_loss.avg:<8.5f} |"
        f"Time (s) : {time.time()-start:<7.4f} |"
        f"Group Sizes: {out_net['group_sizes'].tolist()}")

    return train_loss.avg, train_bpp_loss.avg, train_mse_loss.avg, train_psnr_avg.avg, train_ssim_avg.avg


def test_epoch(epoch, validation_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss            = AverageMeter()
    bpp_loss        = AverageMeter()
    y_bpp_loss      = AverageMeter()
    z_bpp_loss      = AverageMeter()
    mse_loss        = AverageMeter()
    aux_loss        = AverageMeter()
    test_psnr_avg   = AverageMeter()
    test_ssim_avg   = AverageMeter()

    with torch.no_grad():
        for i, data in enumerate(validation_dataloader):
            data         = data['gt_pol'].to(device)
            if args.primary_pol == "HH":
                gt_sar   = data[:,0:2,:,:]
            elif args.primary_pol == "HV":
                gt_sar   = data[:,2:4,:,:]
            elif args.primary_pol == "VH":
                gt_sar   = data[:,4:6,:,:]
            elif args.primary_pol == "VV":
                gt_sar   = data[:,6:8,:,:]        
            gt_sar       = gt_sar.to(device)
            d_dct        = dct.dct_2d(gt_sar)
            out_net = model(d_dct)
            out_criterion = criterion(out_net, gt_sar)

            aux_loss.update(model.aux_loss().item())
            bpp_loss.update(out_criterion["bpp_loss"].item())
            y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
            z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
            loss.update(out_criterion["loss"].item())
            mse_loss.update(out_criterion["mse_loss"].item())
            test_psnr_avg.update(PSNR(out_net["x_hat"], gt_sar).item())
            test_ssim_avg.update(SSIM(out_net["x_hat"], gt_sar).item())

    print(f"Lambda: {args.lmbda} | losstype: {args.loss}| "
        f"Test epoch {epoch}: Average losses:"
        f"Loss: {loss.avg:<7.3f} |"
        f"MSE loss: {mse_loss.avg:<7.3f} |"
        f"Bpp loss: {bpp_loss.avg:<7.4f} |"
        f"y_Bpp loss: {y_bpp_loss.avg:<7.4f} |"
        f"z_Bpp loss: {z_bpp_loss.avg:<7.4f} |"
        f"Aux loss: {aux_loss.avg:<7.4f} |"
        f"Group Sizes: {out_net['group_sizes'].tolist()}\n")

    return loss.avg, bpp_loss.avg, mse_loss.avg, test_psnr_avg.avg, test_ssim_avg.avg


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

class CosineAnnealparameter():
    """
    Cosine annealing scheduler for lambda parameter
    """
    def __init__(self, max, min, T_max):
        
        self.max = max # max value of lambda
        self.min = min # end value of lambda
        self.T_max = T_max # number of epochs
        self.T_cur = 0 # current epoch

    def step(self):
        self.T_cur += 1
        return self.min + 0.5 * (self.max - self.min) * (
            1 + math.cos(math.pi * (self.T_cur / self.T_max)))

    def get_last_lr(self):
        return [self.step()]


def main():
    if args.dataset == 'NGA':
        train_dataset = SarIQDataset(args.train_dataset, train=True)
        validation_dataset  = SarIQDataset(args.validation_dataset, train=False)           

    train_dataloader  = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True)

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False)
    
    net = SAREliC(
        N=args.N,
        M=args.M,
        input_channels=args.inputchannels,
    )
    net = net.to(device)

    if not os.path.exists(args.checkpoint):
        try:
            os.mkdir(args.checkpoint)
        except:
            os.makedirs(args.checkpoint)
    
    writer = SummaryWriter(args.checkpoint)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = RateDistortionLoss(lmbda=args.lmbda, loss=args.loss)
    test_criterion = RateDistortionLoss(lmbda=args.lmbda, loss=args.loss)

    last_epoch = 0
    if args.pretrain:  # load from previous checkpoint
        print("Loading", args.pretrain)
        pretrain = torch.load(args.pretrain, map_location=device)
        last_epoch = pretrain["epoch"] + 1
        net.load_state_dict(pretrain["state_dict"])
        optimizer.load_state_dict(pretrain["optimizer"])
        aux_optimizer.load_state_dict(pretrain["aux_optimizer"])
        lr_scheduler.load_state_dict(pretrain["lr_scheduler"])


    noisequant = 'noise'
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        if epoch >= args.finetune_epoch:
            noisequant = 'ste' # apply STE normalization to latent parameters to finetune
        print("noisequant: {}, {}".format(False if noisequant is None else True, noisequant))
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_loss, train_bpp, train_mse, train_psnr, train_ssim = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            noisequant,
        )
        writer.add_scalar('Train/loss', train_loss, epoch)
        writer.add_scalar('Train/mse', train_mse, epoch)
        writer.add_scalar('Train/bpp', train_bpp, epoch)
        writer.add_scalar('Train/psnr', train_psnr, epoch)
        writer.add_scalar('Train/ssim', train_ssim, epoch)

        loss, bpp, mse, test_psnr, test_ssim = test_epoch(epoch, validation_dataloader, net, test_criterion)
        writer.add_scalar('Test/loss', loss, epoch)
        writer.add_scalar('Test/mse', mse, epoch)
        writer.add_scalar('Test/bpp', bpp, epoch)
        writer.add_scalar('Test/psnr', test_psnr, epoch)
        writer.add_scalar('Test/ssim', test_ssim, epoch)
        
        lr_scheduler.step()

        # add lr plots
        writer.add_scalar('Train/lr', lr_scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('Train/aux_lr', aux_optimizer.param_groups[0]['lr'], epoch)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            # Update entropy bottleneck parameters 
            net.update(force=True)

            DelfileList(args.checkpoint, "checkpoint_last")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                filename=os.path.join(args.checkpoint, "checkpoint_last_{}.pth.tar".format(epoch))
            )
            if is_best:
                DelfileList(args.checkpoint, "checkpoint_best")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    filename=os.path.join(args.checkpoint, "checkpoint_best_loss_{}.pth.tar".format(epoch))
                )

if __name__ == "__main__":
    main()
