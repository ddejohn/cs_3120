import cv2
import pathlib
import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report as crep
from itertools import product
from tabulate import tabulate
from IPython.display import display, Markdown


DEFAULT = {
    "n_neighbors": 3,
    "metric": "manhattan",
    "weights": "distance",
    "n_jobs": -1
}


class Model(KNeighborsClassifier):
    """A container for a KNN classifier"""
    def __init__(self, path: str):
        super().__init__(**DEFAULT)
        self.dims = (32,32)
        self.labels = {}
        self.path = path
        self.parts = ["train", "test", "validate"]
        self.load_data()
        self.fit(self.train.X, self.train.Y)
    # end

    def part_print(self):
        partitions = [self.train, self.test, self.validate]
        n = sum(len(p) for p in partitions)
        for i, p in enumerate(partitions):
            print(f"\nPartition {i}: '{self.parts[i]}'")
            print(f"       size: {len(p)} / {n}")
            print(f"       pcnt: {100*len(p)/n} %")
        # end
    # end

    def load_data(self) -> tuple:
        """Load all .jpg files from all subdirectories in 'path' 
        and then preprocess, label, and partition the data."""
        label_num = {}
        data_set = pathlib.Path(self.path)
        data = []

        # create the label lookup dict for verifcation later
        for i, v in enumerate(data_set.iterdir()):
            label_num[v.name] = i
            self.labels[i] = v.name
        # end

        # read images
        for img_path in data_set.rglob("*.jpg"):
            lbl = label_num[str(img_path.parent.stem)]
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, self.dims, interpolation=cv2.INTER_AREA)

            # flatten RGB data into a vector
            # NOTE: NOT ACTUALLY NECESSARY! 
            img.flatten()

            # label the sample and append to temp data list
            sample = np.append(lbl, img)
            data.append(sample)
        # end

        # partition and package the data (*_ ensures safe unpacking)
        train, test, validate, *_ = Data.partition(data, self.parts, 0.7, 0.2)
        self.train = Data(train)
        self.test = Data(test)
        self.validate = Data(validate)
    # end

    def retrain(self, kwargs=DEFAULT, dims=(32,32)):
        """Retrain the KNN model with parameters from 'kwargs'."""
        super().__init__(**kwargs)

        if dims != self.dims:
            self.dims = dims
            self.load_data()

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

        print(f"\n{report}")
    # end

    def cycle_params(self):
        print("*** WARNING ***")
        print("THIS IS GOING TO TAKE A LONG TIME")

        table = []
        header = ["neighbors", "metric", "weights", "dims", "accuracy"]

        neighbors = [3, 5, 7, 9]
        metrics = ["manhattan", "euclidean", "minkowski"]
        weights = ["uniform", "distance"]
        dims = [(16,16), (32,32), (64,64)]

        for p in product(neighbors, metrics, weights, dims):
            n, m, w, d = p
            params = {
                "n_neighbors": n,
                "metric": m,
                "weights": w,
                "n_jobs": -1
            }
            self.retrain(kwargs=params, dims=d)
            score = self.score(self.test.X, self.test.Y)
            table.append([n, m, w, d, round(score, 5)])
        # end

        table = sorted(table, key=lambda x: x[4])
        table = tabulate(table, header, tablefmt="github")
        display(Markdown(table))
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

    def __len__(self):
        return len(self.Y)
    # end

    @staticmethod
    def partition(data: list, parts: list, *args: float) -> list:
        """Shuffle 'data' and then partition by 'args' proportions.

        Automatically creates a final partition if sum(args) != 1.0"""
        random.seed(42)
        partition_names = parts
        random.shuffle(data)
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
