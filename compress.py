import compressai
import csv
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

from compressai.zoo import load_state_dict
from dct_fast import ImageDCT
from option import args
from ELICUtilis.models.NetworkDCT_v2 import SAREliC
from pathlib import Path
from sar_evaluation_metrics import *
from torch.utils.data import Dataset

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


def compress():
    print("\n\nCompression running on: ", device)
    print("Loading models...")
    compressai.set_entropy_coder(args.entropy_coder)
    data       = torch.load(args.test_model,  map_location=device)
    state_dict = load_state_dict(data['state_dict'])
    model_cls  = SAREliC(N=args.N, M=args.M, input_channels=args.inputchannels, num_slices=5, groups=[0,16,16,32,64,192]).to(device)
    model_cls.load_state_dict(state_dict)
    model      = model_cls.eval()

    # load dataset
    test_dataset = SarIQDataset(args.test_image, args)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # create output directory
    if args.save_encoded is not None:
        os.makedirs(args.save_encoded, exist_ok=True)
        csv_path = os.path.join(args.save_encoded, 'results.csv')

    # with open(csv_path, 'w',) as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["SAR File", "Encoded File", "Bitrate", "Encoding Time"])
        
    with torch.no_grad():
        data          = next(iter(test_loader))
        im_name       = data['name']
        header_file   = os.path.join(args.save_encoded, os.path.splitext(os.path.basename(im_name[0]))[0] + '_header.txt')
        metadata_file = os.path.join(args.save_encoded, os.path.splitext(os.path.basename(im_name[0]))[0] + '_metadata.txt')
        with open(header_file, 'wb') as file_header:
            file_header.write(data["sar_header"][0])
        with open(metadata_file, 'wb') as file_metadata:
            file_metadata.write(data["sar_metadata"][0])
            
        sar_image    = data['gt_pol']
        _ = model.compress(torch.ones(1, 32, 64, 64).to(device))  # warmup
        b, c, h, w   = sar_image.shape
        ps           = 64

        # Calculate padding required along each dimension
        pad_y        = (ps - (h % ps)) % ps
        pad_x        = (ps - (w % ps)) % ps

        # Pad the image using reflection (mirror) padding
        padding      = (0, pad_x, 0, pad_y)  # (left, right, top, bottom)
        gt_sar       = F.pad(sar_image, padding, mode='reflect')

        if 0:
            # Calculate the number of patches in each dimension
            _,_,padded_height, padded_width= gt_sar.shape
            num_patches_y = padded_height // ps
            num_patches_x = padded_width // ps
            # Extract patches
            patches = torch.empty((num_patches_y * num_patches_x, c, ps, ps))
            for i in range(num_patches_y):
                for j in range(num_patches_x):
                    y_start = i * ps
                    y_end = y_start + ps
                    x_start = j * ps
                    x_end = x_start + ps
                    patches[i * num_patches_x + j] = gt_sar[:, :, y_start:y_end, x_start:x_end]
            gt_sar = patches
        print("Compressing...")
        start_time = time.time() 
        image_dct = dct.dct_2d(gt_sar).to(device)
        out_enc = model.compress(image_dct)
        enc_time = time.time() - start_time
        print("Saving bitstream...")
        encode_path = os.path.join(args.save_encoded, os.path.splitext(os.path.basename(im_name[0]))[0] + '.bin')
        
        # save the encoded files
        if args.save_encoded is not None:
            with open(encode_path, 'wb') as file:
                if 1:
                    pickle.dump({"sar": out_enc['strings'],
                                "sar_header": data["sar_header"],
                                "sar_metadata": data["sar_metadata"],
                                "hyper_latent_shape": out_enc["shape"],
                                "pad_x": pad_x,
                                "pad_y": pad_y}, file)
                else:
                    pickle.dump(out_enc['strings'], file)
        
        # Calculate bitrate
        f_size_sarelic = Path(encode_path)
        bpp = f_size_sarelic.stat().st_size * 8.0 / (sar_image.size(0) * sar_image.size(1) * sar_image.size(2) * sar_image.size(3))
        print("SAR File: %s\nEncoded file: %s \nBitrate: %.4f (%d bytes) bpp\nEncoding time: %.4f sec" % (im_name[0], encode_path, bpp, f_size_sarelic.stat().st_size, enc_time))
        #    writer.writerow([im_name, encode_path, bpp, enc_time])

if __name__ == '__main__':
    compress()
    print("Done...")
