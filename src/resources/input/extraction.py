import glob
import cv2
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

seed = 10
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)


all_mask_fn = glob.glob('./masks/*')
mask_df = pd.DataFrame()
mask_df['file_names'] = all_mask_fn
mask_df['mask_percentage'] = 0
mask_df.set_index('file_names',inplace=True)
for fn in all_mask_fn:
    mask_df.loc[fn,'mask_percentage'] = np.array(Image.open(fn)).sum()/(256*256*255) #255 is bcz img range is 255

mask_df.reset_index(inplace=True)
mask_df['labels'] = 0
mask_df.loc[mask_df.mask_percentage>0,'labels'] = 1

all_train_fn = glob.glob('./train/*')
total_samples = len(all_train_fn)
idx = np.arange(total_samples)
train_fn,val_fn = train_test_split(all_train_fn,stratify=mask_df.labels,test_size=0.3,random_state=10)

masks_train_fn = [fn.replace('./train','./masks') for fn in train_fn]
masks_val_fn = [fn.replace('./train','./masks') for fn in val_fn]

for file_path in train_fn:
    x = np.array(Image.open(file_path))
    cv2.imwrite('./keras_im_train/'+file_path.split('/')[-1], x)
for file_path in masks_train_fn:
    x = np.array(Image.open(file_path))
    cv2.imwrite('./keras_mask_train/'+file_path.split('/')[-1], x)
for file_path in val_fn:
    x = np.array(Image.open(file_path))
    cv2.imwrite('./keras_im_val/'+file_path.split('/')[-1], x)
for file_path in masks_val_fn:
    x = np.array(Image.open(file_path))
    cv2.imwrite('./keras_mask_val/'+file_path.split('/')[-1], x)
