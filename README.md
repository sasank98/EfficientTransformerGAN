# Efficient Transformer GAN

Travel Dates: 01

Team members: Ajaysriram Muthuraman & Sasank Potluri, Date: 09/22/2023, CS 7180 Advanced Perception



## Overview

Changed the Generator architecture from ResNet to ETGAN the difference can be seen in the model architectures in figure directory also check the models.ipynb file for implementation to visualize the graphs. Changed the train_net and train_gan files to implement ETGAN as their generator model. the following commands should still work. Please note that the test.py and inference.py file were not changed yet so they will still have the old generator

Implemented a Transformer based architecture for the task of Super-Resolution, used HFM modules to process high frequency information and low frequency information seperately.

Implemented two paths in the neural network one for understanding the scene of the image and other to add minor details to each feature

Below is the architecture of SRGAN implementation

<span align="center"><img width="240" height="360" src="figure/SrGAN_model.svg"/></span>

Below is the architecture of ESRT implementation based on which our model is hugely based on

<span align="center"><img width="240" height="360" src="figure/ESRT_model.svg"/></span>

Finally the architecture of our model 

<span align="center"><img width="240" height="360" src="figure/changed_model.svg"/></span>

From the above images it can be understood that our model looks close to ESRT, we added more convolutional layers in path2 for the model to learn minor features and gave a weightage parameter which can be learned for both paths

We would still be training our model similar to SR-GAN, which is first we train the model with L2 loss function and then followed by Perceptual loss with the help of a discriminator that was trained to find whether an image is a High-Resolution or a Super-Resolution image

<!-- ## Table of contents

- [SRGAN-PyTorch](#srgan-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test SRGAN_x4](#test-srganx4)
        - [Train srresnet_x4](#train-srresnetx4)
        - [Resume train srresnet_x4](#resume-train-srresnetx4)
        - [Train srgan_x4](#train-srganx4)
        - [Resume train srgan_x4](#resume-train-srganx4)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](#photo-realistic-single-image-super-resolution-using-a-generative-adversarial-network) -->

<!-- ## Download weights

Download all available model weights.

```shell
# Download `SRGAN_x4-SRGAN_ImageNet.pth.tar` weights to `./results/pretrained_models`
$ bash ./scripts/download_weights.sh SRGAN_x4-SRGAN_ImageNet
# Download `SRResNet_x4-SRGAN_ImageNet.pth.tar` weights to `./results/pretrained_models`
$ bash ./scripts/download_weights.sh SRResNet_x4-SRGAN_ImageNet
# Download `DiscriminatorForVGG_x4-SRGAN_ImageNet.pth.tar` weights to `./results/pretrained_models`
$ bash ./scripts/download_weights.sh DiscriminatorForVGG_x4-SRGAN_ImageNet
``` -->

## Download datasets

These train images are randomly selected from the verification part of the ImageNet2012 classification dataset.

```shell
$ bash ./scripts/download_datasets.sh SRGAN_ImageNet
```

It is convenient to download some commonly used test data sets here.

```shell
$ bash ./scripts/download_datasets.sh Set5
```

## Install Dependencies

We will be training and testing the model in a conda virtual environment

```shell
$ bash conda create -n [YOUR_ENV_NAME]
```
Install the dependencies using the Yaml File

```shell
$ bash conda install -f TrGAN_requirements.yaml
```

It is convenient to download some commonly used test data sets here.

```shell
$ bash ./scripts/download_datasets.sh Set5

## How Test and Train

Both training and testing only need to modify yaml file.

Set5 is used as the test benchmark in the project, and you can modify it by yourself.

If you need to test the effect of the model, download the test dataset.

```shell
$ bash ./scripts/download_datasets.sh Set5
```

### Test ETGAN_x4

```shell
$ python3 test.py --config_path ./configs/test/SRGAN_x4-SRGAN_ImageNet-Set5.yaml
```

### Test ET_net_x4

```shell
$ python3 test.py --config_path ./configs/test/SRResNet_x4-SRGAN_ImageNet-Set5.yaml
```

### Train ET_net_x4

First, the dataset image is split into several small images to reduce IO and keep the batch image size uniform.

```shell
$ python3 ./scripts/split_images.py
```

Then, run the following commands to train the model

```shell
$ python3 train_net.py --config_path ./configs/train/SRResNet_x4-SRGAN_ImageNet.yaml
```

### Resume train ET_net_x4

Modify the `./configs/train/SRResNet_x4-SRGAN_ImageNet.yaml` file.

- line 33: `RESUMED_G_MODEL` change to `./samples/SRResNet_x4-SRGAN_ImageNet/g_epoch_xxx.pth.tar`.

```shell
$ python3 train_net.py --config_path ./configs/train/SRResNet_x4-SRGAN_ImageNet.yaml
```

### Train ETGAN_x4

```shell
$ python3 train_gan.py --config_path ./configs/train/SRGAN_x4-SRGAN_ImageNet.yaml
```

### Resume train ETGAN_x4

Modify the `./configs/train/SRGAN_x4-SRGAN_ImageNet.yaml` file.

- line 38: `PRETRAINED_G_MODEL` change to `./results/SRResNet_x4-SRGAN_ImageNet/g_last.pth.tar`.
- line 40: `RESUMED_G_MODEL` change to `./samples/SRGAN_x4-SRGAN_ImageNet/g_epoch_xxx.pth.tar`.
- line 41: `RESUMED_D_MODEL` change to `./samples/SRGAN_x4-SRGAN_ImageNet/d_epoch_xxx.pth.tar`.

```shell
$ python3 train_gan.py --config_path ./configs/train/SRGAN_x4-SRGAN_ImageNet.yaml
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1609.04802v5.pdf](https://arxiv.org/pdf/1609.04802v5.pdf)

In the following table, the psnr value in `()` indicates the result of the project, and `-` indicates no test.

| Set5 | Scale |      EffTrNet      |       ETGAN        |
|:----:|:-----:|:------------------:|:------------------:|
| PSNR |   4   |  8.940(**9.340**)  |    8.39(**8.62**)  |
| SSIM |   4   | 0.0182(**0.0210**) | 0.0138(**0.0152**) |

<!-- | Set14 | Scale |      SRResNet      |       SRGAN        |
|:-----:|:-----:|:------------------:|:------------------:|
| PSNR  |   4   |  28.49(**28.57**)  |  26.02(**27.12**)  |
| SSIM  |   4   | 0.8184(**0.7815**) | 0.7397(**0.7321**) |

| BSD100 | Scale |      SRResNet      |       SRGAN        |
|:------:|:-----:|:------------------:|:------------------:|
|  PSNR  |   4   |  27.58(**27.56**)  |  25.16(**26.22**)  |
|  SSIM  |   4   | 0.7620(**0.7367**) | 0.6688(**0.6867**) | -->

```bash
# If you do not train the model yourself, you can download the model weights and test them.
$ bash ./scripts/download_weights.sh SRGAN_x4-SRGAN_ImageNet
$ python3 ./inference.py
```

Input:

<span align="center"><img width="240" height="360" src="figure/comic.png"/></span>

Output:

<span align="center"><img width="240" height="360" src="figure/sr_comic.png"/></span>

<!-- ```text
Build `srresnet_x4` model successfully.
Load `srresnet_x4` model weights `SRGAN-PyTorch/results/pretrained_models/SRGAN_x4-SRGAN_ImageNet.pth.tar` successfully.
SR image save to `./figure/sr_comic.png`
``` -->

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

## Credit

### Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

_Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan
Wang, Wenzhe Shi_ <br>

<!-- **Abstract** <br>
Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central
problem remains largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of
optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on
minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking
high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this
paper, we present SRGAN, a generative adversarial network (GAN) for image super-resolution (SR). To our knowledge, it is the first framework capable
of inferring photo-realistic natural images for 4x upscaling factors. To achieve this, we propose a perceptual loss function which consists of an
adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is
trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by
perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily
downsampled images on public benchmarks. An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SRGAN.
The MOS scores obtained with SRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art method. -->

[[Paper]](https://arxiv.org/pdf/1609.04802v5.pdf)
[[Paper]](https://arxiv.org/pdf/2108.11084.pdf)

Note: this Repo is hugely benifits from https://github.com/Lornatang/SRGAN-PyTorch and https://github.com/louisfghbvc/Efficient-Transformer-for-Single-Image-Super-Resolution

```bibtex
@InProceedings{srgan,
    author = {Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi},
    title = {Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network},
    booktitle = {arXiv},
    year = {2016}
}
```
