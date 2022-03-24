import cv2
import numpy as np
import os

for root, subdir, files in os.walk("background/"):
    for filename in files:
        print("Processing : ", filename)
        img = cv2.imread(os.path.join(root, filename))
        bigger = cv2.resize(img, (200, 200))
        cv2.imwrite(os.path.join(root, filename), bigger)
        print("Saved!")
