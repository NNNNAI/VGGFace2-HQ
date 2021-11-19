'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-15 19:42:42
LastEditors: Naiyuan liu
LastEditTime: 2021-11-19 16:17:54
Description: 
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse
import cv2
import glob
from tqdm import tqdm
from insightface_func.face_detect_crop_ffhq_newarcAlign import Face_detect_crop
import argparse

def align_image_dir(dir_name_tmp):
    ori_path_tmp = os.path.join(input_dir, dir_name_tmp)
    image_filenames = glob.glob(os.path.join(ori_path_tmp,'*'))
    save_dir_newarcalign = os.path.join(output_dir_newarcalign,dir_name_tmp)
    save_dir_ffhqalign = os.path.join(output_dir_ffhqalign,dir_name_tmp)
    if not os.path.exists(save_dir_newarcalign):
        os.makedirs(save_dir_newarcalign)
    if not os.path.exists(save_dir_ffhqalign):
        os.makedirs(save_dir_ffhqalign)


    for file in image_filenames:
        image_file = os.path.basename(file)

        image_file_name_newarcalign = os.path.join(save_dir_newarcalign, image_file)
        image_file_name_ffhqalign  = os.path.join(save_dir_ffhqalign, image_file)
        if os.path.exists(image_file_name_newarcalign) and os.path.exists(image_file_name_ffhqalign):
            continue

        face_img = cv2.imread(file)
        if face_img.shape[0]<250 or face_img.shape[1]<250:
            continue    
        ret = app.get(face_img,crop_size,mode=mode)
        if len(ret)!=0 :
            cv2.imwrite(image_file_name_ffhqalign, ret[0])
            cv2.imwrite(image_file_name_newarcalign, ret[1])
        else:
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir',type=str,default = '/home/gdp/harddisk/Data1/VGGface2/train')
    parser.add_argument('--output_dir_ffhqalign',type=str,default = '/home/gdp/harddisk/Data1/VGGface2_ffhq_align')
    parser.add_argument('--output_dir_newarcalign',type=str,default = '/home/gdp/harddisk/Data1/VGGface2_newarc_align')
    parser.add_argument('--crop_size',type=int,default = 256)
    parser.add_argument('--mode',type=str,default = 'Both')

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir_newarcalign = args.output_dir_newarcalign
    output_dir_ffhqalign = args.output_dir_ffhqalign
    crop_size = args.crop_size
    mode      = args.mode

    app = Face_detect_crop(name='antelope', root='./insightface_func/models')

    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(320,320))

    dirs = sorted(os.listdir(input_dir))
    handle_dir_list = dirs
    for handle_dir_list_tmp  in tqdm(handle_dir_list):
        align_image_dir(handle_dir_list_tmp)

