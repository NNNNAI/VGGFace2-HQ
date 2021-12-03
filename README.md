# VGGFace2-HQ

## The first open source high resolution dataset for face swapping!!!

A high resolution version of [VGGFace2](https://github.com/ox-vgg/vgg_face2) for academic face editing purpose.This project uses [GFPGAN](https://github.com/TencentARC/GFPGAN) for image restoration and [insightface](https://github.com/deepinsight/insightface) for data preprocessing (crop and align).

[![logo](./VGGFace2-HQ.png)](https://github.com/NNNNAI/VGGFace2-HQ)

We provide a download link for users to download the data, and also provide guidance on how to generate the VGGFace2 dataset from scratch.

If you find this project useful, please star it. It is the greatest appreciation of our work.

<img src="./docs/img/vggface2_hq_compare.png"/>

# Get the VGGFace2-HQ dataset from cloud!

We have uploaded the dataset of VGGFace2 HQ to the cloud, and you can download it from the cloud.

### Google Drive

[[Google Drive]](https://drive.google.com/drive/folders/1ZHy7jrd6cGb2lUa4qYugXe41G_Ef9Ibw?usp=sharing)

***We are especially grateful to [Kairui Feng](https://scholar.google.com.hk/citations?user=4N5hE8YAAAAJ&hl=zh-CN) PhD student from Princeton University.***

### Baidu Drive

[[Baidu Drive]](https://pan.baidu.com/s/1LwPFhgbdBj5AeoPTXgoqDw) Password: ```sjtu```


# Generate the HQ dataset by yourself. (If you want to do so)
## Preparation
### Installation
**We highly recommand that you use Anaconda for Installation**
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install insightface==0.2.1 onnxruntime
(optional) pip install onnxruntime-gpu==1.2.0

pip install basicsr
pip install facexlib
pip install -r requirements.txt
python setup.py develop
```
- The pytorch and cuda versions above are most recommanded. They may vary.
- Using insightface with different versions is not recommanded. Please use this specific version.
- These settings are tested valid on both Windows and Ununtu.
### Pretrained model
- We use the face detection and alignment methods from **[insightface](https://github.com/deepinsight/insightface)** for image preprocessing. Please download the relative files and unzip them to **./insightface_func/models** from [this link](https://onedrive.live.com/?authkey=%21ADJ0aAOSsc90neY&cid=4A83B6B633B029CC&id=4A83B6B633B029CC%215837&parId=4A83B6B633B029CC%215834&action=locate).
- Download [GFPGANCleanv1-NoCE-C2.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth) from GFPGAN offical repo. Place "GFPGANCleanv1-NoCE-C2.pth" in **./experiments/pretrained_models**.

### Data preparation
- Download VGGFace2 Dataset from [VGGFace2 Dataset for Face Recognition](https://github.com/ox-vgg/vgg_face2)

## Inference

- Frist, perform data preprocessing on all photos in VGGFACE2, that is, detect faces and align them to the same alignment format as FFHQdataset.
```
python scripts/crop_align_vggface2_FFHQalign.py --input_dir $DATAPATH$/VGGface2/train --output_dir_ffhqalign $ALIGN_OUTDIR$ --mode ffhq --crop_size 256
```
- And then, do the magic of image restoration with GFPGAN for processed photos.
```
python scripts/inference_gfpgan_forvggface2.py --input_path $ALIGN_OUTDIR$  --batchSize 8 --save_dir $HQ_OUTDIR$
```

## Related Projects

***Please visit our popular face swapping project***

[![logo](./docs/img/simswap.png)](https://github.com/neuralchen/SimSwap)

***Please visit our another ACMMM2020 high-quality style transfer project***

[![logo](./docs/img/logo.png)](https://github.com/neuralchen/ASMAGAN)

[![title](/docs/img/title.png)](https://github.com/neuralchen/ASMAGAN)

***Please visit our AAAI2021 sketch based rendering project***

[![logo](./docs/img/girl2.gif)](https://github.com/TZYSJTU/Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale)
[![title](/docs/img/girl2-RGB.png)](https://github.com/TZYSJTU/Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale)

Learn about our other projects 

[[VGGFace2-HQ]](https://github.com/NNNNAI/VGGFace2-HQ);

[[RainNet]](https://neuralchen.github.io/RainNet);

[[Sketch Generation]](https://github.com/TZYSJTU/Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale);

[[CooGAN]](https://github.com/neuralchen/CooGAN);

[[Knowledge Style Transfer]](https://github.com/AceSix/Knowledge_Transfer);

[[SimSwap]](https://github.com/neuralchen/SimSwap);

[[ASMA-GAN]](https://github.com/neuralchen/ASMAGAN);

[[SNGAN-Projection-pytorch]](https://github.com/neuralchen/SNGAN_Projection)

[[Pretrained_VGG19]](https://github.com/neuralchen/Pretrained_VGG19).



# Acknowledgements

<!--ts-->
* [GFPGAN](https://github.com/TencentARC/GFPGAN)
* [Insightface](https://github.com/deepinsight/insightface)
* [VGGFace2 Dataset for Face Recognition](https://github.com/ox-vgg/vgg_face2)
<!--te-->
