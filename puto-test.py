import io
import cv2
import base64
import requests


from PIL import Image


# A1111 URL
url = "http://0.0.0.0:7861"

# Read Image in RGB order
img = cv2.imread('/Users/luisguajardo/Desktop/sem-explorations-Auto1111/automatic1111-automationTest/tests/06.png')[:, :, ::-1]

# Encode into PNG and send to ControlNet
retval, bytes = cv2.imencode('.png', img)
encoded_image = base64.b64encode(bytes).decode('utf-8')

# A1111 payload
payload = {
    "prompt": 'a handsome man',
    "negative_prompt": "",
    "batch_size": 1,
    "steps": 20,
    "cfg_scale": 7,
    "alwayson_scripts": {
        "controlnet": {
            "args": [
                {
                    "input_image": encoded_image,
                    "module": "canny",
                    "model": "control_v11p_sd15_canny [d14c016b]",
                }
            ]
        }
    }
}


# Trigger Generation
response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

# Read results
r = response.json()
result = r['images'][0]
image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
image.save('output.png')