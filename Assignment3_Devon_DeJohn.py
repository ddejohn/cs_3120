import pathlib
import numpy as np
import pandas as pd
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


class AnimalData:
    def __init__(self, path):
        self.image_set = []
        self.label_set = []
        self.label_dct = {}
        self.load_data(path)

    def load_data(self, path):
        label_num = {}
        data_set = pathlib.Path(path)

        for i, v in enumerate(data_set.iterdir()):
            label_num[v.name] = i
            self.label_dct[i] = v.name

        for img_path in data_set.rglob("*.jpg"):
            print(img_path,)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)
            lbl = img_path.parent.stem
            self.image_set.append(img)
            self.label_set.append(label_num[lbl])

# end

#     def load(self, path):
#         with os.scandir(data_path) as images:
#             for label_num, class_type in enumerate(images):
#                 label_dct[label_num] = class_type.name
#                 sub_dir = f"{data_path}/{class_type.name}"
#                 with os.scandir(sub_dir) as class_samples:
#                     for sample in class_samples:
#                         img_path = f"{sub_dir}/{sample.name}"
#                         img = cv2.imread(img_path)
#                         img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)
#                         image_set.append(img)
#                         label_set.append(label_num)


# end

animal_data = AnimalData("datasets/animals")
print(animal_data.label_dct,)
