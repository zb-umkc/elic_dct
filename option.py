import argparse
import compressai
import os
import time

home_path = os.getenv("HOME")

# Training settings
parser = argparse.ArgumentParser(description="SAREliC compression")
parser.add_argument("--mode", type=str, default="test",
                    help="train or test (default: %(default)s)")
parser.add_argument("--primary_pol", type=str, default='HH',
                    help="Primary polarization to use `VV` or `HH` or `VH` or `HV` (default: %(default)s)")
parser.add_argument("--test_image", type=str, default='/media/paras/WD_BLACK/PythonDir/dataset/SAR_dataset/NGA/raw/sicd_example_2_PFA_RE32F_IM32F_HH.nitf', #validation_NGA_multi_pol_ps256', #  
                    help="Test image path (.nift)")
parser.add_argument("--save_encoded", default="/media/paras/WD_BLACK/PythonDir/SAREliC-Compression/encodedbinaries/",
                    help="save the encoded files, path to the directory or None to not save data")
parser.add_argument("--test_model", type=str, default="/media/paras/WD_BLACK/PythonDir/SAREliC-Compression/checkpoint-NGA-DCTv2/checkpoint_best_lambda4.pth.tar",
                    help="checkpoint path during testing")
parser.add_argument("--dataset", type=str, default='NGA', 
                    help='Dataloader to use `NGA`or `Sandia` or `JPL` % (default: %(default)s)')
parser.add_argument("--inputchannels", default=2, type=int, 
                    help="Number of input channels (default: %(default)s)")
parser.add_argument("--min_val", type=float, default=-5000, 
                    help="Minimum value of SAR image (default: %(default)s)")
parser.add_argument("--max_val", type=float, default=5000,
                    help="Maximum value of SAR image (default: %(default)s)")
parser.add_argument("--N", type=int, default=192, 
                    help="Number of channels of main codec (default: %(default)s)")
parser.add_argument("--M", type=int, default=320,
                    help="Number of channels of latent codec (default: %(default)s)")
parser.add_argument("-n", "--num-workers", type=int, default=8,
                    help="Dataloaders threads (default: %(default)s)")
parser.add_argument("--patch_size", type=int, nargs=2, default=(1024, 1024),
                    help="Size of the patches to be cropped (default: %(default)s)")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Use cuda")
parser.add_argument("--save", action="store_true", default=True, 
                    help="Save model to disk")
parser.add_argument("--seed", type=float, default=1926, 
                    help="Set random seed for reproducibility")
parser.add_argument('--gpu_id', type=str, default='0',
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("-c", "--entropy-coder", choices=compressai.available_entropy_coders(), default=compressai.available_entropy_coders()[0],
    help="entropy coder (default: %(default)s)", # currently only ans supported in compressAI
)

args = parser.parse_args()
assert os.path.exists(args.test_image), "test image path not found"
