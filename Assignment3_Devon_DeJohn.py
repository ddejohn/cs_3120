import cv2
import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from random import shuffle


class ImageKNN:
    def __init__(self, path):
        self.labels = {}
        self.train, self.test, self.validate = self.load_data(path)

    def load_data(self, path):
        label_num = {}
        data_set = pathlib.Path(path)
        raw_data = []

        # create the label lookup dict for verifcation later
        for i, v in enumerate(data_set.iterdir()):
            label_num[v.name] = i
            self.labels[i] = v.name

        # read images
        for img_path in data_set.rglob("*.jpg"):
            lbl = label_num[str(img_path.parent.stem)]
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)

            # label the sample and append to temp data list
            sample = np.append(lbl, img.flatten())
            raw_data.append(sample)

        # partition and package the data (*_ ensures safe unpacking)
        train, test, validate, *_ = Data.partition(raw_data, 0.7, 0.1)
        return Data(train), Data(test), Data(validate)
# end


class Data:
    """A basic data structure"""
    def __init__(self, data: list):
        self.X = []
        self.Y = []
        for d in data:
            self.Y.append(d[0])
            self.X.append(d[1:])
    
    @staticmethod
    def flatten(data: list):
        """Recursively flatten data into a 1D list"""
        data = sum(data, [])
        if not isinstance(data[0], list):
            return data
        return Data.flatten(data)

    @staticmethod
    def partition(data: list, *args: float):
        """Shuffle 'data', then partition by the proportions given in 'args'

        Automatically creates a final partition if sum(args) != 1.0
        """
        shuffle(data)
        n = len(data)
        parts = []
        rem, a, b = n, 0, 0
        for p in args:
            b = a + int(n*p)
            parts.append(data[a:b])
            rem -= (b - a)
            a = b
        parts.append(data[-rem:])
        return parts
# end




animal_data = ImageKNN("datasets/animals")
# print(animal_data.label_keys,)
