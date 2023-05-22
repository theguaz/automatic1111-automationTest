import json
import base64
import random
import string
import time
import requests
import base64
import io

from PIL import Image

noUI_URL = "http://127.0.0.1:7861/sdapi/v1/img2img"
API_URL = "http://0.0.0.0:7861/sdapi/v1/img2img"

testArray = ["09","07","06","04", "10"]
testFolder = "/Users/luisguajardo/Desktop/sem-explorations-Auto1111/automatic1111-automationTest/tests/"

def generate_random_string(length=6):
    characters = string.digits
    random_string = ''.join(random.choice(characters) for i in range(length))
    return random_string

def submit_post(url: str, data: dict):
    return requests.post(url, data=json.dumps(data))


def save_encoded_image(b64_image: str, output_path: str):
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(b64_image))

def getBaseImage (imageName):
    with Image.open(testFolder + imageName + ".png") as img:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()

        # Convert bytes to base64 string
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_b64

def getMaskImage (imageName):
    with Image.open(testFolder + imageName + "_mask.png") as img:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        img_b64M = base64.b64encode(img_bytes).decode('utf-8')
        return img_b64M



def generateImage():
    whichImage = testArray[3]#random.choice(testArray)
    txt2img_url = API_URL
    
    baseImage = str( getBaseImage(whichImage) ) 
    maskImage = str( getMaskImage(whichImage) )

    start_time = time.time()
    

    print("generating image...")

    data = {
            "init_images":[ baseImage ],
            "mask": maskImage,
            "prompt": "A realistic highly detailed photograph of (((1 PERSON ONLY ))) (((1 woman ))) POSING ((( IN A RUSSIAN THEME PARK WITH BEARS))) for a prom photography session dressed in very fancy party-like formal clothing, UHD, 8k, Kodak lenses, nice lighting, highly detailed, press photo, high resolution, hyper realistic, ambient lighting, Nikon D850, 50mm f/1.8 lens, formal prom outfits posing for a picture, (((vibrant backdrops)))",
            "negative_prompt":"Negative prompt: 3D,illustration,sketch,drawing, low quality,deformed,malformed,ugly, oversaturated, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render,crop",
            "steps": 40,
            "mask_mode":"inpaint_not_masked",
            "sampler_name": "DPM++ SDE Karras",
            "cfg_scale":10,
            "width":960,
            "height":512,
            "model_hash":"c6bbc15e32",
            "denoise_strength":0.75,
            "resize_mode": 0,
            "mask_blur": 12, 
            
            "inpainting_fill": 1,
            "inpaint_full_res": False,
             "inpainting_mask_invert": 1,
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "enabled": True,
                            "input_image": baseImage,
                            "module": "openpose",
                            "mask": "",
                            "model": "control_openpose-fp16 [9ca67cc5]",
                            "weight": 1.5,
                            "width":960,
                            "height":512,
                            "resize_mode": 1,
                            "low_vram": False,
                            "processor_res": 512,
                            "threshold_a": 0,
                            "threshold_b": 1,
                            "guidance_start": 0, 
                            "guidance_end": 1, 
                            "control_mode": 1,
                            "pixel_perfect": True
                        }
                    ]
                }
            }
           
           

            }
    response = submit_post(txt2img_url, data)
   
    end_time = time.time()

    # Calculate the elapsed time and print it
    elapsed_time = end_time - start_time
    print(" image generated :) in -->" + str(elapsed_time) +" secs")
    save_encoded_image(response.json()['images'][0], 'test_python/test_img2img_' + str(elapsed_time) +"-secs_" + generate_random_string() + '.png')


for _ in range(20): 
    generateImage()