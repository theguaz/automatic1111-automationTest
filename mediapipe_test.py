import json
import base64
import random
import string
import time
import requests
import base64
import io
import numpy as np
import cv2
import sys
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_filenames_in_dir(directory):
    return [f for f in os.listdir(directory) 
            if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]



model_path = "./selfie_multiclass_256x256.tflite"
directory = "./test_v2/"  # replace with your directory
filenames = get_filenames_in_dir(directory)



BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
print(str( mp.tasks.vision))
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a image segmenter instance with the image mode:
options = ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=model_path), running_mode=VisionRunningMode.IMAGE,  output_confidence_masks = True,output_category_mask = False,)


def savefaceSegmented(file_name):
        with ImageSegmenter.create_from_options(options) as segmenter:
            
            imgName_toSave = os.path.splitext(file_name)[0] + "_mask.png"
            print(imgName_toSave)

            BG_COLOR = (0, 0, 0) # gray
            MASK_COLOR = (255, 255, 255) # white
            mp_image = mp.Image.create_from_file(directory + file_name)
            segmented_masks = segmenter.segment(mp_image)
            #get face segmentation from multiclass, The model outputs the following segmentation categories: ``` 0 - background 1 - hair 2 - body-skin 3 - face-skin 4 - clothes 5 - others (accessories) ```
            category_mask = segmented_masks.confidence_masks[3]
            # Generate solid color images for showing the output segmentation mask.
            image_data = mp_image.numpy_view()
            fg_image = np.zeros(image_data.shape, dtype=np.uint8)
            fg_image[:] = MASK_COLOR
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
            output_image = np.where(condition, fg_image, bg_image)
            window_name = 'image'
          
            ksize = (10,10)
            blurredIMG = cv2.blur(output_image, ksize) 
            cv2.imwrite(directory + imgName_toSave, blurredIMG)
            

for filename in filenames:
    savefaceSegmented(filename)


print("executed succesfully")