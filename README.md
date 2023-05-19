# AI Image Generation Project

This project leverages the automatic 1111 API to generate AI images and uses img2ig and masks to create unique images. We have reached a significant milestone by successfully creating 100 unique images.

In order to run the python script you need to run Automatic1111 with the flag --api and --nowebui
This will run the local server as usual but all the requests go to a different url, the server when ready will show something like this:

`INFO:     Uvicorn running on http://127.0.0.1:7861 (Press CTRL+C to quit)`


## Results

You can find the generated images and the Python code example that was used in this project at the following locations:

- test_automate_automatic.py is the file currently generating images
- folder named test_python contains generated results

## Ongoing Work

Currently, I'm are looking into integrating PoseNet into our requests. It appears that PoseNet may not be significantly influencing the image generation process as intended. We aim to troubleshoot and fully utilize PoseNet's capabilities to further enhance our results.

## Contribute

We welcome contributions, ideas, and suggestions to improve the quality and efficiency of our image generation. Please feel free to explore the code and images, and share your insights.
