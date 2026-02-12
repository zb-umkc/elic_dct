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
parser.add_argument("--lambda", dest="lmbda", type=float, default=4,
                    help="Bit-rate distortion parameter (default: %(default)s)")
parser.add_argument("--loss", type=int, default=1,
                    help = "1: MSE, 2: L1, 3: I/Q loss, 4: NMSE loss, 5: Corr Loss (default: %(default)s)")
parser.add_argument("--train-dataset", type=str, default='/scratch/zb7df/data/NGA/multi_pol/train', 
                    help="Training dataset path")
parser.add_argument("--validation-dataset", type=str, default='/scratch/zb7df/data/NGA/multi_pol/train_val', 
                    help="Validation dataset path")
parser.add_argument("--test-dataset", type=str, default='/scratch/zb7df/data/NGA/multi_pol/validation',  
                    help="Test dataset path")
parser.add_argument('--checkpoint', type=str, default='/scratch/zb7df/checkpoints/elic_dct/',
                    help='Path to save the checkpoint, use different path for different experiments')
parser.add_argument("--save_encoded", default="./encodedbinaries/",
                    help="save the encoded files, path to the directory or None to not save data")
parser.add_argument("--test-model", type=str, default="/home/pmc4p/PythonDir/SAREliC-Compression-master/checkpoint-NGA-DCTv2/checkpoint_best_lambda4.pth.tar",
                    help="checkpoint path during testing")
parser.add_argument("--dataset", type=str, default='NGA', 
                    help='Dataloader to use `NGA`or `Sandia` or `JPL` % (default: %(default)s)')
parser.add_argument("--datatype", type=str, default='IQ', 
                    help='ONLY if using `SAR_amp` datasetType. -> Data type to use `amp` (for amplitude) or `I` (for inphase) or `Q` (for quadrature) or `IQ` (for both)')
parser.add_argument("--pretrain", type=str, default=None, 
                    help="Path to a pretrain")
parser.add_argument("--clip_max_norm", type=float, default=1.0, 
                    help="gradient clipping max norm (default: %(default)s)")
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
parser.add_argument("-e", "--epochs", type=int, default=250,
                    help="Number of epochs (default: %(default)s)")
parser.add_argument("--finetune_epoch", type=int, default=175,
                    help="Epoch to start finetuning or applying STE normalization (default: %(default)s)")
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4,
                    help="Learning rate (default: %(default)s)")
parser.add_argument("--min_lr", default=0.0, type=float, 
                    help="Minimum learning rate for cosine annealing scheduler (default: %(default)s)") 
parser.add_argument("--aux-learning-rate", type=float, default=1e-3,
                    help="Auxiliary loss learning rate (default: %(default)s)")
parser.add_argument("-n", "--num-workers", type=int, default=8,
                    help="Dataloaders threads (default: %(default)s)")
parser.add_argument("--batch-size", type=int, default=16, 
                    help="Batch size (default: %(default)s)")
parser.add_argument("--test-batch-size", type=int, default=1,
                    help="Test batch size (default: %(default)s)")
parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256),
                    help="Size of the patches to be cropped (default: %(default)s)")
parser.add_argument("--cuda", action="store_true", default=True,
                    help="Use cuda")
parser.add_argument("--save", action="store_true", default=True, 
                    help="Save model to disk")
parser.add_argument("--seed", type=float, default=1926, 
                    help="Set random seed for reproducibility")
parser.add_argument('--gpu-id', type=str, default='0',
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("-c", "--entropy-coder", choices=compressai.available_entropy_coders(), default=compressai.available_entropy_coders()[0],
    help="entropy coder (default: %(default)s)", # currently only ans supported in compressAI
)
# parser.add_argument("--bypass", action="store_true", default=False,
#                     help="Bypass rANS and context model using RLE/Exp-Golomb",)
parser.add_argument('--bypass-grps', type=str, default='0',
                    help='Channel groups to bypass using RLE/Exp-Golomb, ex. 134')

args = parser.parse_args()

if args.mode == "train":
    args.train_dataset = os.path.join(home_path, args.train_dataset)
    assert os.path.exists(args.train_dataset), "Training dataset path not found"
    args.validation_dataset = os.path.join(home_path, args.validation_dataset)
    assert os.path.exists(args.validation_dataset), "Validation dataset path not found"
    args.checkpoint = os.path.join(home_path, args.checkpoint)
    #assert os.path.exists(args.checkpoint), "Checkpoint path not found"
    args.checkpoint = os.path.join(args.checkpoint, "%s_DCT_grpsizes_M%d_lmbda%.1f"%(args.dataset, args.M, args.lmbda))
    if os.path.exists(args.checkpoint):
        print("Checkpoint path already exists")
        args.checkpoint = os.path.join(args.checkpoint, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.checkpoint, exist_ok=True)
    if args.pretrain is not None:
        args.pretrain = os.path.join(home_path, args.pretrain)
        assert os.path.exists(args.pretrain), "Pretrain path not found"
    # save hyperparameters to same location as the model
    with open(os.path.join(args.checkpoint, "hyperparameters.txt"), "w") as f:
        for key, value in args._get_kwargs():
            f.write(f"{key}: {value}\n")
else:
    args.test_dataset = os.path.join(home_path, args.test_dataset)
    assert os.path.exists(args.test_dataset), "test dataset path not found"
    args.patch_size = (256, 256)
    # assert os.path.exists(args.test_model), "Test model path not found"
