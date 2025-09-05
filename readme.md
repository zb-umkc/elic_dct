# SAR End-to-End Compression

## Dataset
We used [National Geospatial-intelligence Agency (NGA) dataset](https://umkc.box.com/s/203foyzrx2xt94w69qkujp35k7vab41d) to train, test and validate our work. The NGA dataset consists of 8 complex-valued SAR images with two instances of the same scene and four different polarizations (HH, HV, VH, and VV). Our current results are based on the HH polarization only. 

The first scene is used for training our network. The second image is used for testing and validating out network.

## Install anaconda environment

## Conda env
'''
conda create --name neurcom python=3.10
conda activate neurcom
# Mac
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
# Linux/Windows CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# Linux/Windows CPU only
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

'''

### Train 
'''
python train_NGA_DCTv2.py --mode train --train_dataset TRAIN_PATH --validation_dataset VALIDATION_PATH --test_dataset TEST_PATH --batch_size 16
'''

### Test 
'''
python train_NGA_DCTv2.py --mode test --test_dataset TEST_PATH --test_model TEST_MODEL_PATH
'''