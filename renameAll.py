import os

# Specify the directory you want to rename files in
folder_path = '/Users/luisguajardo/Desktop/sem-explorations-Auto1111/automatic1111-automationTest/test_v2'

# Filter all image files in the directory
img_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

for i, filename in enumerate(sorted(img_files), start=1):
    # Get the extension
    ext = os.path.splitext(filename)[1]
    
    # Form the new name
    new_name = str(i).zfill(2) + ext  # It will add leading zero if i < 10

    # Rename the file
    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))

print("All image files have been renamed.")
