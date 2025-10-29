import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class NMSELoss(nn.Module):
    def __init__(self, args):
        super(NMSELoss, self).__init__()
        self.min_val = args.min_val
        self.max_val = args.max_val
        self.amp_max_val = torch.sqrt(torch.tensor(self.max_val ** 2 + self.min_val ** 2))
    def forward(self, y, x):
        # y: predicted sar image range [-5000, 5000]
        # x: target sar image range [-5000, 5000]
        # convert into complex number
        y = y[0,0,:,:] + 1j*y[0,1,:,:]
        x = x[0,0,:,:] + 1j*x[0,1,:,:]

        y_x = y - x
        # y_x conjugate
        y_x = torch.real(y_x*torch.conj(y_x))
        # x conjugate
        xx = torch.real(x*torch.conj(x))
        # y conjugate
        yy = torch.real(y*torch.conj(y))
        nmse = torch.mean(y_x)/(torch.sqrt(torch.mean(xx)) * torch.sqrt(torch.mean(yy)))
        #plt.imshow(y_x/torch.sqrt(xx*yy));plt.colorbar()
        return nmse

class CorrLoss(nn.Module):
    def __init__(self, args):
        super(CorrLoss, self).__init__()
        self.min_val = args.min_val
        self.max_val = args.max_val
        self.amp_max_val = torch.sqrt(torch.tensor(self.max_val ** 2 + self.min_val ** 2))
    def forward(self, y, x):
        # y: predicted sar image range [-5000, 5000]
        # x: target sar image range [-5000, 5000]
        # convert into complex number
        y = y[0,0,:,:] + 1j*y[0,1,:,:]
        x = x[0,0,:,:] + 1j*x[0,1,:,:]

        # yx*
        yx = torch.real(y*torch.conj(x))
        # x mag square
        x2 = torch.real(np.abs(x)**2)
        # y mag square
        y2 = torch.real(np.abs(y)**2)
        # Figure of Merit or Pearson correlation coefficient
        corr = torch.mean(yx)/(torch.sqrt(torch.mean(x2) * torch.mean(y2)))
        #plt.imshow(yx/torch.sqrt(x2*y2));plt.colorbar()
        return corr

if __name__ == "__main__":
    from option import args
    from torch.utils.data import DataLoader
    from ELICUtilis.datasets import SarAmpDataset

    min_val = args.min_val
    max_val = args.max_val
    args.patch_size = (1024, 1024)
    n_bits  = 12
    amp_max_val = torch.sqrt(torch.tensor(max_val ** 2 + min_val ** 2))
    
    test_data   = SarAmpDataset(args.testroot, train=False, data_type=args.datatype)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    img = next(iter(test_loader))
    img = img * (max_val - min_val) + min_val
    nmse_loss = NMSELoss(args)
    print(nmse_loss(img, img))
    corr_loss = CorrLoss(args)
    print(corr_loss(img, img))
    print("Done...")

