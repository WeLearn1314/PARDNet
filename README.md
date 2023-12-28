# PARDNet
Recovering a clean background: A parallel deep network architecture for single-image deraining (PARDNet) by Nanrun Zhou, Jibin Deng and Meng Pang*, Pattern Recognition Letters (Major Revision), 2023.

# Prerequisites:

python == 3.6.2

tensorflow == 2.0.0

keras == 2.3.1

opencv-python == 4.5.5.62

scikit-image == 0.17.2

# Training
For train the Rain100H, please run:

python train.py --image_dir_noise you rain data --image_dir_original you gt data --test_dir_noise you test rain data --test_dir_original you test gt data --If_n True

For train the Rain100L, please run:

python train.py --image_dir_noise you rain data --image_dir_original you gt data --test_dir_noise you test rain data --test_dir_original you test gt data --If_n False

# Testing
For test the Rain100H, please run:

python test.py --If_n True

For test the Rain100L, please run:

python test.py --If_n False

The dataset "Rain100H" and "Rain100L" you can download here:

https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html