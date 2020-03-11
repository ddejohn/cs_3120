import cv2
import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class ImageData:
    def __init__(self, path):
        self.imgs = []
        self.labels = []
        self.label_keys = {}
        self.train = Data()
        self.test = Data()
        self.validation = Data()
        self.load_data(path)
        self.reshape_data()

    def load_data(self, path):
        label_num = {}
        data_set = pathlib.Path(path)

        for i, v in enumerate(data_set.iterdir()):
            label_num[v.name] = i
            self.label_keys[i] = v.name

        for img_path in data_set.rglob("*.jpg"):
            lbl = img_path.parent.stem
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)
            self.imgs.append(img)
            self.labels.append(label_num[lbl])
    
    def reshape_data(self):
        data = np.array(self.imgs)
        data = data.reshape(data.shape[0], 3072)
        self.imgs = data

    def train_test_validate(self):
        trainX, testX, trainY, testY = train_test_split(
            self.imgs, self.labels, train_size=0.7)
        self.train.X = trainX
        self.train.Y = trainY
        
        tX, tY, vX, vY = train_test_split(testX, testY, train_size=0.2)
        
        self.test.X = testX
        self.test.Y = testY

# end



class Data:
    def __init__(self):
        self.X = []
        self.Y = []


# animal_data = ImageData("datasets/animals")
# print(animal_data.label_keys,)



def flatten(data):
    return sum(data, [])



def partition(data, *args):
    n = len(data) 
    partitioned_data = []
    p1 = 0
    p2 = 0
    rem = n
    for p in args:
        p2 = p1 + int(n*p)
        rem -= (p2 - p1)
        part = data[p1:p2]
        p1 = p2
        partitioned_data.append(part)
    print(rem)
    partitioned_data.append(data[-rem:])
    return partitioned_data


x = [*range(1237)]
partition(x, *(0.4, 0.3, 0.2, 0.1))
