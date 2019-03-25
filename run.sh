## 3月13日
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 20
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 32
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 44
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 56

#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 20 --dataset CIFAR100
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 32 --dataset CIFAR100
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 44 --dataset CIFAR100
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 56 --dataset CIFAR100

## 3月15日
#CUDA_VISIBLE_DEVICES=1 python -u train.py --arch densenet --depth 100 --block_type bottleneck --growth_rate 12 --compression_rate 0.5 --batch_size 32 --base_lr 0.05 --seed 7 --dataset CIFAR100
#CUDA_VISIBLE_DEVICES=1 python -u train.py --arch pyramidnet --depth 110 --block_type basic --pyramid_alpha 84 --seed 7 --dataset CIFAR100
#CUDA_VISIBLE_DEVICES=1 python -u train.py --arch shake_shake --depth 26 --base_channels 32 --shake_forward True --shake_backward True --shake_image True --seed 7 --dataset CIFAR100
#CUDA_VISIBLE_DEVICES=1 python -u train.py --arch shake_shake --depth 26 --base_channels 64 --shake_forward True --shake_backward True --shake_image True --batch_size 64 --base_lr 0.1 --seed 7 --dataset CIFAR100

#CUDA_VISIBLE_DEVICES=1 python -u train.py --arch densenet --depth 100 --block_type bottleneck --growth_rate 12 --compression_rate 0.5 --batch_size 32 --base_lr 0.05 --seed 7
#CUDA_VISIBLE_DEVICES=1 python -u train.py --arch pyramidnet --depth 110 --block_type basic --pyramid_alpha 84 --seed 7
#CUDA_VISIBLE_DEVICES=1 python -u train.py --arch shake_shake --depth 26 --base_channels 32 --shake_forward True --shake_backward True --shake_image True --seed 7
#CUDA_VISIBLE_DEVICES=1 python -u train.py --arch shake_shake --depth 26 --base_channels 64 --shake_forward True --shake_backward True --shake_image True --batch_size 64 --base_lr 0.1 --seed 7

## 3月20日
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 20 --use_label_smoothing --label_smoothing_epsilon 0.01
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 32 --use_label_smoothing --label_smoothing_epsilon 0.01
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 44 --use_label_smoothing --label_smoothing_epsilon 0.01
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 56 --use_label_smoothing --label_smoothing_epsilon 0.01

#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 20 --use_label_smoothing --label_smoothing_epsilon 0.01 --dataset CIFAR100
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 32 --use_label_smoothing --label_smoothing_epsilon 0.01 --dataset CIFAR100
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 44 --use_label_smoothing --label_smoothing_epsilon 0.01 --dataset CIFAR100
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 56 --use_label_smoothing --label_smoothing_epsilon 0.01 --dataset CIFAR100


## 3月21日
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 20 --use_mixup
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 32 --use_mixup
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 44 --use_mixup
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 56 --use_mixup

# 3月22日
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 20 --use_mixup  --dataset CIFAR100
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 32 --use_mixup  --dataset CIFAR100
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 44 --use_mixup  --dataset CIFAR100
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 56 --use_mixup  --dataset CIFAR100

#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 20 --use_random_erasing
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 32 --use_random_erasing
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 44 --use_random_erasing
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 56 --use_random_erasing

#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 20 --use_random_erasing  --dataset CIFAR100
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 32 --use_random_erasing  --dataset CIFAR100
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 44 --use_random_erasing  --dataset CIFAR100
#CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 56 --use_random_erasing  --dataset CIFAR100

# 3月23日
CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 20 --scheduler cosine
CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 32 --scheduler cosine
CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 44 --scheduler cosine
CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 56 --scheduler cosine

CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 20 --scheduler cosine  --dataset CIFAR100
CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 32 --scheduler cosine  --dataset CIFAR100
CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 44 --scheduler cosine  --dataset CIFAR100
CUDA_VISIBLE_DEVICES=1 python train.py --arch resnet --depth 56 --scheduler cosine  --dataset CIFAR100

