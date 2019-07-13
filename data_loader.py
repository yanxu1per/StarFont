from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import pdb

class ImageFolder(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, transform,mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.train_images = list(map(lambda x: os.path.join(image_dir+mode, x), os.listdir(image_dir+mode)))
        self.transform = transform
        self.num_images = len(self.train_images)
        self.mode=mode
        
    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        
        #random.seed()
        #random.shuffle(self.train_images)

        src = self.train_images[index]
        print(index)

        src_char = int(src.split('_')[0][len(self.image_dir+self.mode+'/'):])
        src_style = int(src.split('_')[1][:-len(".jpg")])
        #pdb.set_trace()
        try:
            trg = random.choice([x for x in self.train_images
                                    if '_'+str(src_style) in x and str(src_char)+'_' not in x])
            #print(1)
        except:
            trg = src
            #print(2)
        trg_style = int(trg.split('_')[1][:-len(".jpg")])
        trg_char = int(trg.split('_')[0][len(self.image_dir+self.mode+'/'):])
        src = self.transform(Image.open(src))
        trg = self.transform(Image.open(trg))

        return src, src_style, src_char, \
               trg, trg_style, trg_char

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class ImageFolder1(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, transform,mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.train_imagefs = list(map(lambda x: os.path.join(image_dir+mode, x), os.listdir(image_dir+mode)))
        self.transform = transform
        self.num_images = len(self.train_imagefs)
        self.mode=mode
        
    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        if self.mode=='train':
            random.seed()
            random.shuffle(self.train_imagefs)
        #print(index)

        fo = self.train_imagefs[index]
        imgs=os.listdir(fo)
        x=[]
        for item in imgs:
            x.append(self.transform(Image.open(fo+'/'+item)))
        x=torch.cat(x, dim=0)
        
        return x

    def __len__(self):
        """Return the number of images."""
        return self.num_images

def get_loader(m,image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    transform.append(T.ToTensor())
    #transform.append(T.Normalize(mean=[0.5], std=[0.5]))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        if m==10:
            dataset = ImageFolder1(image_dir, transform,mode)
        else:
            dataset = ImageFolder(image_dir, transform,mode)
    
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader