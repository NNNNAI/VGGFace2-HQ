# VGGFace2-HQ
A high resolution version of [VGGFace2](https://github.com/ox-vgg/vgg_face2) for face editing purpose.This project uses [GFPGAN](https://github.com/TencentARC/GFPGAN) for image restoration and [insightface](https://github.com/deepinsight/insightface) for data preprocessing (crop and align).

# Preparation
## Installation
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
## Pretrained model
- We use the face detection and alignment methods from **[insightface](https://github.com/deepinsight/insightface)** for image preprocessing. Please download the relative files and unzip them to **./insightface_func/models** from [this link](https://onedrive.live.com/?authkey=%21ADJ0aAOSsc90neY&cid=4A83B6B633B029CC&id=4A83B6B633B029CC%215837&parId=4A83B6B633B029CC%215834&action=locate).
- Download [GFPGANCleanv1-NoCE-C2.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth) from GFPGAN offical repo. Place "GFPGANCleanv1-NoCE-C2.pth" in **./experiments/pretrained_models**.

## Data preparation
- Download VGGFace2 Dataset from [VGGFace2 Dataset for Face Recognition](https://github.com/ox-vgg/vgg_face2)

# Inference

- Frist, perform data preprocessing on all photos in VGGFACE2, that is, detect faces and align them to the same alignment format as FFHQdataset.
```
python scripts/crop_align_vggface2_FFHQalign.py --input_dir $DATAPATH$/VGGface2/train --output_dir_ffhqalign $ALIGN_OUTDIR$ --mode ffhq --crop_size 256
```
- And then, do the magic of image restoration with GFPGAN for processed photos.
```
python scripts/inference_gfpgan_forvggface2.py --input_path $ALIGN_OUTDIR$  --batchSize 8 --save_dir $HQ_OUTDIR$
```
# Acknowledgements

<!--ts-->
* [GFPGAN](https://github.com/TencentARC/GFPGAN)
* [Insightface](https://github.com/deepinsight/insightface)
* [VGGFace2 Dataset for Face Recognition](https://github.com/ox-vgg/vgg_face2)
<!--te-->
