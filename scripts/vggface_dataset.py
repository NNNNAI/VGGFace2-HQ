'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-16 19:32:18
LastEditors: Naiyuan liu
LastEditTime: 2021-11-16 19:35:12
Description:
'''
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import glob



class TotalDataset(data.Dataset):

    def __init__(self,image_dir,
                    content_transform):
        self.image_dir     = image_dir

        self.content_transform= content_transform
        self.dataset = []
        self.mean = [0.5, 0.5, 0.5]
        self.std  = [0.5, 0.5, 0.5]
        self.preprocess()
        self.num_images = len(self.dataset)

    def preprocess(self):
        additional_pattern = '*/*'
        self.dataset.extend(sorted(glob.glob(os.path.join(self.image_dir, additional_pattern), recursive=False)))

        print('Finished preprocessing the VGGFACE2 dataset...')


    def __getitem__(self, index):
        """Return single image."""
        dataset = self.dataset

        src_filename1 = dataset[index]

        src_image1          = self.content_transform(Image.open(src_filename1))


        return src_image1, src_filename1


    def __len__(self):
        """Return the number of images."""
        return self.num_images

def getLoader(c_image_dir, ResizeSize=512, batch_size=16, num_workers=8):
    """Build and return a data loader."""
    c_transforms = []


    c_transforms.append(T.Resize([ResizeSize,ResizeSize]))
    c_transforms.append(T.ToTensor())
    c_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    c_transforms = T.Compose(c_transforms)

    content_dataset = TotalDataset(c_image_dir, c_transforms)


    sampler = None
    content_data_loader = data.DataLoader(dataset=content_dataset,batch_size=batch_size,
                    drop_last=False,num_workers=num_workers,sampler=sampler,pin_memory=True)
    return content_data_loader

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

