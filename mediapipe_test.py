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


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



model_path = "/Users/luisguajardo/Desktop/sem-explorations-Auto1111/automatic1111-automationTest/selfie_multiclass_256x256.tflite"
img_test = "/Users/luisguajardo/Desktop/sem-explorations-Auto1111/automatic1111-automationTest/tests/09.png"

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
print(str( mp.tasks.vision))
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def save_encoded_image(b64_image: str, output_path: str):
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(b64_image))


# Create a image segmenter instance with the image mode:
options = ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=model_path), running_mode=VisionRunningMode.IMAGE, output_category_mask=True)
with ImageSegmenter.create_from_options(options) as segmenter:
    BG_COLOR = (0, 0, 0) # gray
    MASK_COLOR = (255, 255, 255) # white

    mp_image = mp.Image.create_from_file(img_test)

    segmented_masks = segmenter.segment(mp_image)
    category_mask = segmented_masks.category_mask
    print(str(segmented_masks))
    # Generate solid color images for showing the output segmentation mask.
    image_data = mp_image.numpy_view()
    
    fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR

    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2

    output_image = np.where(condition, fg_image, bg_image)
    
    window_name = 'image'
  
    cv2.imshow(window_name, output_image)
      
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)
      
    # closing all open windows
    cv2.destroyAllWindows()

    #save_encoded_image(str(mp_image), "test.png")
   

print("executed succesfully")