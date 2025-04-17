"""
This snippet processes the CUB dataset to generate train/test .txt files with the following content

<img_path>,<class_labels>,<top_x>,<top_y>,<btm_x>,<btm_y>,<312 (processed)attribute labels>

"""

import pickle
import os
from glob import glob

cubp_f = open("cubp_test.txt", "w")
for root, dirs, files in os.walk("G:\DATA\DomainAdaptation\CUB\CUB-200-Painting\images"):
    for dir in dirs:
        image_names = [dir + "/" +x for x in os.listdir(os.path.join(root,dir)) if x[-4:]==".jpg"]
        # labels = [int(image_name[:3]) for image_name in image_names]
        for image_name in image_names:
            cubp_f.write(image_name + "," + f"{int(image_name[:3])}"+",0,0,0,0" + "\n")
cubp_f.close()
        # image_names = [image_name for image_name in image_names]
        # res_dict[dir[4:].replace("_",' ').lower()] = image_names

