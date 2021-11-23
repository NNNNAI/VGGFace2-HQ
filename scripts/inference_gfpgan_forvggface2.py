'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-16 19:30:52
LastEditors: Naiyuan liu
LastEditTime: 2021-11-20 17:57:55
Description:
'''
import os
import torch
import argparse

from tqdm import tqdm
from vggface_dataset import getLoader
from basicsr.utils import imwrite, tensor2img
from gfpgan.archs.gfpganv1_arch import GFPGANv1
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
import platform

def main_worker(args):
    arch = args.arch

    with torch.no_grad():
        # initialize the GFP-GAN
        if arch == 'clean':
            gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=args.channel,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        else:
            gfpgan = GFPGANv1(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=args.channel,
                decoder_load_path=None,
                fix_decoder=True,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)

        loadnet = torch.load(args.model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        gfpgan.load_state_dict(loadnet[keyname], strict=True)
        gfpgan.eval()
        gfpgan.cuda()

        test_dataloader = getLoader(args.input_path, 512, args.batchSize, 8)

        print(len(test_dataloader))


        for images,filenames in tqdm(test_dataloader):
            images = images.cuda()

            output_batch = gfpgan(images, return_rgb=False)[0]

            for tmp_index in range(len(output_batch)):
                tmp_filename = filenames[tmp_index]

                split_leave = tmp_filename.split(args.input_path)[-1].split(split_name)
                restored_face = output_batch[tmp_index]
                restored_face = tensor2img(restored_face, rgb2bgr=True, min_max=(-1, 1))
                restored_face = restored_face.astype('uint8')

                sub_dir = os.path.join(args.save_dir, split_leave[-2])
                os.makedirs(sub_dir, exist_ok=True)

                save_path_tmp = os.path.join(sub_dir, split_leave[-1])

                imwrite(restored_face, save_path_tmp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', type=str, default='clean')
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth')
    parser.add_argument('--input_path', type=str, default='/Data/VGGface2_FFHQalign')
    parser.add_argument('--sft_half', default = False, action='store_true')
    parser.add_argument('--batchSize', type=int, default = 8)
    parser.add_argument('--save_dir', type=str, default = ' ')
    parser.add_argument('--channel', type=int, default=2)

    args = parser.parse_args()
 
    if platform.system().lower() == 'windows':
        split_name = '\\'
    elif platform.system().lower() == 'linux':
        split_name = '/'
    os.makedirs(args.save_dir, exist_ok=True)
    main_worker(args)
