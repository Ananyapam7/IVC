import os
import cv2

for dirname, _, filenames in os.walk(''): #Enter the directory in which you want to convert the images to greyscale
    for filename in filenames:
        print("Processing : ", filename)
        img = cv2.imread(os.path.join(dirname, filename))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(dirname, filename), gray)
        print(f"{filename} : Saved!")
