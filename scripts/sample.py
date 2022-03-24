# Import required module
import numpy as np
import os

for root, subdirs, files in os.walk(''): # Enter the root directory which contains all the directories with images of each of the classes
    n = 100 # Number of Images you want to keep in each class
    for subdir in subdirs:
        print(subdir)
        arr = os.listdir(os.path.join(root, subdir))
        print(len(arr))
        np.random.shuffle(arr)
        arr2 = arr[:n]
        arr3 = arr[n:]
        for name in arr3:
            os.remove("train/" + os.path.join(subdir, name))
            print(f"{name} removed!")
