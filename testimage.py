import cv2
import numpy as np
import torch

# img=cv2.imread('data/Yingrenshi/images/DJI_0145.JPG')
# print(img.shape)
# tensor=torch.tensor(img).unsqueeze(0).permute(0,3,1,2)
# print(tensor.shape)
# ret=tensor.squeeze(0).permute(1,2,0).numpy()
# print(ret.shape)
# cv2.imwrite("testimage/img1.JPG",ret)

str='/root/repository/project/sci/log/LoG/data/Yingrenshi/cache/4/images/DJI_0381.JPG'
affi=str.split('/')
print(affi[-1])