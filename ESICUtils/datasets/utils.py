import matplotlib.pyplot as plt
import os 
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class SarIQDataset(Dataset):
    def __init__(self, root, train=True, names=False, min_val=-5000, max_val=5000) -> None:
        super().__init__()
        self.train = train
        self.names = names # return name of image as well

        self.min_val = min_val
        self.max_val = max_val
        self.gt_pol1 = os.path.join(root, "gt_HH")
        self.gt_pol2 = os.path.join(root, "gt_HV")
        self.gt_pol3 = os.path.join(root, "gt_VH")
        self.gt_pol4 = os.path.join(root, "gt_VV")
        self.file_list = os.listdir(os.path.abspath(self.gt_pol1))
        
        assert len(self.file_list) > 0, f"No files found in {self.root}"
        
        self.file_list = sorted(list(filter(lambda x: '.npy' in x, self.file_list)))

        self.dataset_size = len(self.file_list)


    def _train_transforms(self, gt_image_pol):
        """
        Random image transforms.
        """
        if random.random() > 0.5:
            gt_image_pol = TF.hflip(gt_image_pol)
        if random.random() > 0.5:
            gt_image_pol = TF.vflip(gt_image_pol)
        if random.random() > 0.5:
            random_rotate = random.choice([-90, 90])
            gt_image_pol = TF.rotate(gt_image_pol, random_rotate)
            
        
        return gt_image_pol

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        input_image_name =  self.file_list[idx]
        # shape: W * H * C
        gt_image_pol1     = np.load(os.path.join(self.gt_pol1, input_image_name)).astype(np.float32)
        gt_image_pol2     = np.load(os.path.join(self.gt_pol2, input_image_name)).astype(np.float32)
        gt_image_pol3     = np.load(os.path.join(self.gt_pol2, input_image_name)).astype(np.float32)
        gt_image_pol4     = np.load(os.path.join(self.gt_pol2, input_image_name)).astype(np.float32)

        # shape: C * W * H
        gt_image_pol1     = np.stack([gt_image_pol1[:,:,0], gt_image_pol1[:,:,1]], axis=0)
        gt_image_pol2     = np.stack([gt_image_pol2[:,:,0], gt_image_pol2[:,:,1]], axis=0)
        gt_image_pol3     = np.stack([gt_image_pol3[:,:,0], gt_image_pol3[:,:,1]], axis=0)
        gt_image_pol4     = np.stack([gt_image_pol4[:,:,0], gt_image_pol4[:,:,1]], axis=0)
 
        gt_image_pol      = np.concatenate([gt_image_pol1, gt_image_pol2, gt_image_pol3, gt_image_pol4], axis=0)
        
        gt_image_pol       = torch.tensor(gt_image_pol, dtype=torch.float32)

        if self.train:
            gt_image_pol = self._train_transforms(gt_image_pol)


        gt_image_pol    = (gt_image_pol - self.min_val) / (self.max_val - self.min_val)

        return {
            'gt_pol': gt_image_pol,
            'name': input_image_name,
        }

if __name__ == '__main__':
    import sys, os
    sys.path.append("/home/pmc4p/PythonDir/SAREliC-Compression-master")
    from torch.utils.data import DataLoader

    test_dataset = "/home/pmc4p/PythonDir/dataset/SAR_dataset/NGA/multi_pol/validation_NGA_multi_pol_ps256/"
    dataset = SarIQDataset(test_dataset, train=True, n_bits=12)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    data = next(iter(dataloader))
    img = data['gt_pol'].cuda()
    plt.figure()
    plt.imshow(img[0,0,:,:].data.cpu().numpy())
    plt.figure()
    plt.imshow(img[0,1,:,:].data.cpu().numpy())
    plt.figure()
    plt.imshow(img[0,2,:,:].data.cpu().numpy())
    plt.figure()
    plt.imshow(img[0,3,:,:].data.cpu().numpy())
    plt.figure()
    plt.imshow(img[0,4,:,:].data.cpu().numpy())
    plt.figure()
    plt.imshow(img[0,5,:,:].data.cpu().numpy())
    plt.figure()
    plt.imshow(img[0,6,:,:].data.cpu().numpy())
    plt.figure()
    plt.imshow(img[0,7,:,:].data.cpu().numpy())
    plt.show()
    print(data['gt_pol'].shape, data['gt_pol'].min(), data['gt_pol'].max())

    print("Done...")