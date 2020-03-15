import cv2
import pathlib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report as crep
from random import shuffle


class ImageKNN(KNeighborsClassifier):
    """A container for a KNN classifier"""
    def __init__(self, path: str, dims=(32,32)):
        super().__init__()
        self.labels = {}
        self.train, self.test, self.validate = self.load_data(path, dims)
        self.fit(self.train.X, self.train.Y)
    # end

    def load_data(self, path: str, dims: tuple) -> tuple:
        """Load all .jpg files from all subdirectories in 'path' 
        and then preprocess, label, and partition the data."""
        label_num = {}
        data_set = pathlib.Path(path)
        raw_data = []

        # create the label lookup dict for verifcation later
        for i, v in enumerate(data_set.iterdir()):
            label_num[v.name] = i
            self.labels[i] = v.name
        # end

        # read images
        for img_path in data_set.rglob("*.jpg"):
            lbl = label_num[str(img_path.parent.stem)]
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, dims, interpolation=cv2.INTER_AREA)

            # label the sample and append to temp data list
            sample = np.append(lbl, img)
            raw_data.append(sample)
        # end

        # partition and package the data (*_ ensures safe unpacking)
        train, test, validate, *_ = Data.partition(raw_data, 0.7, 0.1)
        return Data(train), Data(test), Data(validate)
    # end

    def retrain(self, kwargs):
        """Retrain the KNN model with parameters from 'kwargs'."""
        super().__init__(**kwargs)
        self.fit(self.train.X, self.train.Y)
    # end

    def report(self):
        """Generate a classification report."""
        Y_test = self.test.Y
        Y_predict = self.predict(self.test.X)

        report = crep(
            Y_test, Y_predict,
            labels=[*self.labels.keys()],
            target_names=[*self.labels.values()]
        )

        return f"\n{report}"
    # end
# end


class Data:
    """A basic data structure"""
    def __init__(self, data: list):
        self.X = []
        self.Y = []
        for d in data:
            self.Y.append(d[0])
            self.X.append(d[1:])
        # end
    # end
    
    @staticmethod
    def flatten(data: list) -> list:
        """Recursively flatten 'data' into a 1D list"""
        data = sum(data, [])
        if not isinstance(data[0], list):
            return data
        return Data.flatten(data)
    # end

    @staticmethod
    def partition(data: list, *args: float) -> list:
        """Shuffle 'data' and then partition by 'args' proportions.

        Automatically creates a final partition if sum(args) != 1.0"""
        shuffle(data)
        n = len(data)
        rem, a, b = n, 0, 0
        parts = []

        for p in args:
            b = a + int(n*p)
            parts.append(data[a:b])
            rem -= (b - a)
            a = b
        # end

        parts.append(data[-rem:])
        return parts
    # end
# end


animals = ImageKNN("data/animals")
print(animals.report())

animals.retrain({"p": 2, "n_neighbors": 7, "weights": "distance"})
print(animals.report())
