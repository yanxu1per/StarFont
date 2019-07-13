import util
from PIL import Image
from torchvision import transforms
import cv2
import pdb
import torch
t1=transforms.Compose([transforms.ToTensor()])
def new_read(imgname):
	return t1(cv2.imread(imgname)).view(1,3,128,128)
img0=new_read('0.jpg')
img1=new_read('1.jpg')
img2=new_read('2.jpg')
print(ssim(img0,img1))

