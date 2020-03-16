import cv2
import pathlib
import random
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report as crep
from itertools import product
from tabulate import tabulate


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
        self.dims = (8,8)
        self.labels = {}
        self.path = path
        self.parts = ["train", "test", "validate"]
        self.load_data()
        self.fit(self.train.X, self.train.Y)
    # end

    def datasets(self):
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

    def retrain(self, dims, kwargs=DEFAULT):
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
        # print("*** WARNING: THIS IS GOING TO TAKE A LONG TIME ***")

        table = []
        header = ["neighbors", "metric", "weights", "dims", "accuracy"]

        # dims = [8, 16, 32]
        # neighbors = [3, 5, 7]
        dims = [8, 16, 32, 64]
        neighbors = [3, 5, 7, 9]
        metrics = ["manhattan", "euclidean"]
        weights = ["uniform", "distance"]

        accs = {
            "manhattan": {
                "uniform": [],
                "distance": [],
            },
            "euclidean": {
                "uniform": [],
                "distance": [],
            }
        }

        prod = product(dims, neighbors, metrics, weights)
        card = len(neighbors)*len(metrics)*len(weights)*len(dims)

        for i, p in enumerate(prod, 1):
            # x = int(round(48*i/card, 2))
            # prog = f"[{('x'*x).ljust(48, '.')}]"
            # print(prog, end="\r", flush=True)
            d, n, m, w = p
            params = {
                "n_neighbors": n,
                "metric": m,
                "weights": w,
                "n_jobs": -1
            }
            self.retrain(kwargs=params, dims=(d,d))
            score = round(self.score(self.test.X, self.test.Y), 4)
            accs[m][w].append((dims.index(d), neighbors.index(n), score))
            table.append([n, m, w, (d,d), score])
        # end
        Data.acc_plot(accs)

        table = sorted(table, key=lambda x: x[4])
        print(tabulate(table, header))
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

    @staticmethod
    def acc_plot(data: dict):
        fig = make_subplots(
            rows=2, cols=2,
            row_titles=["manhattan", "euclidean"],
            column_titles=["uniform", "distance"],
            vertical_spacing=0.05,
            horizontal_spacing=0.05,
            x_title="vote weight"
            y_title="metric",
        )

        for i, d in enumerate(data.values(), 1):
            for j, z in enumerate(d.values(), 1):
                contour = Data.make_contour(z)
                fig.add_trace(contour, row=i, col=j)
                fig.update_xaxes(
                    row=i, col=j,
                    tickvals=[8, 16, 32, 64]
                )
                fig.update_yaxes(
                    row=i, col=j,
                    tickvals=[3, 5, 7, 9]
                )
            # end
        # end

        fig.update_xaxes(
            row=2, col=1,
            title_text="dimensions"
        )

        fig.update_yaxes(
            row=2, col=1,
            title_text="neighbors"
        )

        ttl = f"<b>Accuracy vs dimensions vs neighbors</b>"
        fig.update_layout(
            width=950, height=950,
            title=ttl,
        )

        fig.show()
    # end

    @staticmethod
    def make_contour(data: list):
        z = np.zeros((4, 4))
        for tup in data:
            x, y, acc = tup

            # numpy arrays are stored in column order, so switch x and y
            z[y, x] = acc
        # end

        contour = go.Contour(
            z=z, x=[8,16,32,64], y=[3,5,7,9],
            contours=dict(
                showlabels=True,
                labelfont=dict(
                    size=12,
                    color="gray"
                ),
            ),
            colorscale=[
                [0.0, "lightsalmon"],
                [0.5, "gold"],
                [1.0, "mediumturquoise"]
            ],
            showscale=False,
            line_width=0
        )

        return contour
# end
