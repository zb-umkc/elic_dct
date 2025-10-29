import argparse
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sarpy.io.general.nitf as nitf
import scipy.io as sio
import time

from pathlib import Path
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

home_root = os.getenv("HOME")

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Set for visualization")
    parser.add_argument("--data_root", default="/media/paras/WD_BLACK/ood", help="directory for dataset")
    parser.add_argument("--mode", default=["train","test","validation"], help="train, test or validation")
    parser.add_argument("--dataset", default="NGA", help="{NGA, Sandia, UAVSAR}")
    parser.add_argument("--pol", default=["HH", "HV", "VH", "VV"], help="{HH, HV, VH, VV}")
    parser.add_argument("--min_val", default=-5000, help="minimum value of SAR image")
    parser.add_argument("--max_val", default=5000, help="maximum value of SAR image")
    parser.add_argument("--ps", default=256, help="patch size")
    parser.add_argument("--samples", nargs=3, type=int, default=[10, 2, 1], help="number of samples")
    # paths
    parser.add_argument("--input_dir", default="PythonDir/dataset/SAR_dataset/NGA/raw/", help="input dataset path")
    parser.add_argument("--output_dir", default="PythonDir/dataset/SAR_dataset/NGA/multi_pol_", help="output path")
    args = parser.parse_args()

    if args.data_root == None:
        args.data_root = home_root
    args.home_root = home_root
    args.amp_max_val = np.sqrt(args.min_val**2+args.max_val**2)

    args.input_dir = os.path.join(args.data_root, args.input_dir)
    args.output_dir = os.path.join(args.data_root, args.output_dir)

    return args

def main():
    args        = options()
    # import pdb; pdb.set_trace()
    # print all the args in loop
    print("\n","*"*100)
    for arg in vars(args):
        print("{:15}".format(arg), getattr(args, arg))
    print("*"*100, "\n")
    
    for mode in args.mode:
        if mode == 'train':
            file_path  = os.path.join(args.input_dir, "train")
            samples    = args.samples[0]
            ps = args.ps
        if mode == 'validation':
            file_path  = os.path.join(args.input_dir, "test")
            samples    = args.samples[1]
            ps = args.ps
        elif mode == 'test' or mode == 'test2':
            file_path  = os.path.join(args.input_dir, "test")
            samples    = args.samples[2]
            ps = 1024

        output_file_path = os.path.join(args.output_dir, "%s_%s_multi_pol_ps%d/"%(mode, args.dataset, ps))
        # if os.path.exists(output_file_path) == False:
        #     os.makedirs(output_file_path)

        for pol in args.pol:
            if args.dataset == "NGA":
                file_list = glob.glob(os.path.join(file_path, "*%s.nitf"%(pol)))
            else:
                print(args.dataset, "dataset not implemented...")
                return

            for idx in range(len(file_list)):
                input_image_name = file_list[idx]
                if args.dataset == "NGA":
                    sar_data         = nitf.NITFReader(input_image_name)
                    sar_image        = sar_data.read_raw()
                    if args.debug:
                        with open(input_image_name, "rb") as f:
                            sar_header   = f.read(sar_data.nitf_details.img_segment_offsets[0])
                            print("\n****************\n")
                            print(sar_header)
                            print("\n****************\n")

                            f.seek(sar_data.nitf_details.des_subheader_offsets[0])
                            sar_metadata = f.read()
                            print("\n****************\n")
                            print(sar_metadata)
                            print("\n****************\n")
                
                total_pixels = sar_image.size
                clipped_pixels = np.sum((sar_image < args.min_val) | (sar_image > args.max_val))
                clipped_percentage = (clipped_pixels / total_pixels) * 100
                print(f"Percentage of pixels that will be clipped: {clipped_percentage:.2f}%")
                sar_image   = np.clip(sar_image, args.min_val, args.max_val)
                #sar_image   = (sar_image - args.min_val) / (args.max_val - args.min_val)



                H, W, C     = sar_image.shape
                if not os.path.exists(os.path.join(output_file_path, "gt_%s"%(pol))):
                    os.makedirs(os.path.join(output_file_path, "gt_%s"%(pol)))
                # create intput and GT pair
                for j in range(samples):
                    print(mode,pol,idx, j)
                    if mode == "test" or mode == "test2":
                        xx = 0
                        yy = 0
                    else:
                        xx = np.random.randint(0, W - ps)
                        yy = np.random.randint(0, H - ps)
                    
                    gt_patch = sar_image[yy:yy + ps, xx:xx + ps, :]
                    if args.debug:
                        amp_patch = np.sqrt(gt_patch[:,:,0]**2 + gt_patch[:,:,1]**2)
                        plt.imshow(amp_patch, cmap='gray');plt.show()
                    
                    np.save(os.path.join(output_file_path, "gt_%s"%(pol), "%d.npy"%(j+idx*samples)), gt_patch)


if __name__ == "__main__":
    main()

print("Done..")
