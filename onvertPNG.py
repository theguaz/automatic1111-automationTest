import os
from PIL import Image

# Specify the directory you want to convert files in
folder_path = '/Users/luisguajardo/Desktop/sem-explorations-Auto1111/automatic1111-automationTest/test_v2'

for filename in os.listdir(folder_path):
    if filename.endswith('.jpeg') or filename.endswith('.jpg'):
        img = Image.open(os.path.join(folder_path, filename))
        img.save(os.path.join(folder_path, filename.split('.')[0] + '.png'), 'png')
        print(f'{filename} has been converted to png.')
