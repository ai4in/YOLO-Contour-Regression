from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

#数据增强

import os

savepath='/home/data2/mxl/ultralytics/ultralytics-main/corrupted/'
image_folder ='/home/data2/mxl/ultralytics/ultralytics-main/tree_train_split/images/train/'

imagelist=os.listdir(image_folder)


corruptype=["motion_blur",'snow', 'frost', 'fog',
                    'brightness', 'spatter']
i=0
for image in imagelist:
    i=i+1
    print(image_folder+image," :",i)
    for corruption in corruptype:
        # tic = time.time()
        name=image.split(".")[0]
        data = np.asarray(Image.open(image_folder+image).convert('RGB'))
        # print(data.shape)
        if corruption=='spatter' and 'brightness':
            corrupted = corrupt(data, corruption_name=corruption, severity=3)
        else:
            corrupted = corrupt(data, corruption_name=corruption, severity=1)
        # print("corrupted",corrupted.shape)
        Image.fromarray((corrupted)).save('{}/{}_{}.png'.format(savepath,name,corruption))
        # print(corruption, time.time() - tic)