import os
import pathlib
import numpy as np
import pandas as pd
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


class ImageData:
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
            lbl = img_path.parent.stem
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)
            self.image_set.append(img)
            self.label_set.append(label_num[lbl])
# end


animal_data = ImageData("datasets/animals")
print(animal_data.label_dct,)
