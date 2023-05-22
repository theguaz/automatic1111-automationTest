from PIL import Image
import glob

# Specify the directory you wish to use
image_dir = "/Users/luisguajardo/Desktop/sem-explorations-Auto1111/automatic1111-automationTest/test_python/*"

# Use glob to get all the image files from the directory
image_files = glob.glob(image_dir)

# Create a list to hold the images
image_list = []

# Open each image and append it to image_list
for image_file in sorted(image_files):  # sorted to maintain some order, can be removed if not necessary
    img = Image.open(image_file)
    image_list.append(img)

# Create the gif
image_list[0].save('output.gif', save_all=True, append_images=image_list[1:], loop=0, duration=500)
