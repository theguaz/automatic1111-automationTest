# -*- coding: utf-8 -*-
"""Face-Normalization-with-MediaPipe.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/serengil/tensorflow-101/blob/master/python/Face-Normalization-with-MediaPipe.ipynb
"""

import cv2
import matplotlib.pyplot as plt
import mediapipe
import numpy as np
import pandas as pd

img = cv2.imread("pexels-cottonbro-8090149-scaled.jpeg")
#img = cv2.imread("deepface/tests/dataset/img1.jpg")

fig = plt.figure(figsize = (8, 8))
plt.axis('off')
plt.imshow(img[:, :, ::-1])
plt.show()

"""# Facial Landmarks Detector"""

mp_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

landmarks = results.multi_face_landmarks[0]

df = pd.DataFrame(list(mp_face_mesh.FACEMESH_FACE_OVAL), columns = ["p1", "p2"])

df.head()

print(f"Face oval consists of {df.shape[0]} lines")

"""## Order landmark points"""

routes_idx = []

p1 = df.iloc[0]["p1"]
p2 = df.iloc[0]["p2"]

for i in range(0, df.shape[0]):
    
    #print(p1, p2)
    
    obj = df[df["p1"] == p2]
    p1 = obj["p1"].values[0]
    p2 = obj["p2"].values[0]
    
    route_idx = []
    route_idx.append(p1)
    route_idx.append(p2)
    routes_idx.append(route_idx)

routes_idx[0:5]

display_items = 5
for idx, route_idx in enumerate(routes_idx[0:display_items] + routes_idx[-display_items:]):
    print(f"Draw a line between {route_idx[0]}th landmark point to {route_idx[1]}th landmark point")
    if idx == display_items - 1:
        print("\n...\n")

"""## Find the 2D coordinate values of each landmark point"""

routes = []

#for source_idx, target_idx in mp_face_mesh.FACEMESH_FACE_OVAL:
for source_idx, target_idx in routes_idx:
    
    source = landmarks.landmark[source_idx]
    target = landmarks.landmark[target_idx]
        
    relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
    relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))

    #cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness = 2)
    
    routes.append(relative_source)
    routes.append(relative_target)

print(f"There are {len(routes)} landmark points available")

routes[0:10]

"""## Extract the inner area of facial landmarks"""

mask = np.zeros((img.shape[0], img.shape[1]))
mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
mask = mask.astype(bool)
 
out = np.zeros_like(img)
out[mask] = img[mask]

fig = plt.figure(figsize = (15, 15))
plt.axis('off')
plt.imshow(out[:, :, ::-1])

