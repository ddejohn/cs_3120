import csv
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA


class Data:
    """A basic data structure"""
    def __init__(self, data: list):
        self.X = []
        self.Y = []
        for d in data:
            *x, y = d
            self.X.append(x)
            self.Y.append(y)
        # end
    # end

    def __len__(self):
        return len(self.Y)
    # end

    @staticmethod
    def partition(data: list, *args: float) -> list:
        """Shuffle 'data' and then partition by 'args' proportions.

        Automatically creates a final partition if sum(args) != 1.0"""
        n = len(data)
        random.seed(12)
        random.shuffle(data)
        args = [*args, *filter(lambda x: x > 0, [round(1.0-sum(args), 5)])]
        a, b = 0, 0
        parts = []

        for p in args:
            b = a + round(n*p)
            parts.append(data[a:b])
            a = b
        return parts
    # end
# end


class Model:
    def __init__(self, path: str):
        self.labels = {}
        data = self.load_data(path)

        # partition and package the data (*_ ensures safe unpacking)
        train, test, validate, *_ = Data.partition(data, 0.7, 0.2)
        self.train = Data(train)
        self.test = Data(test)
        self.validate = Data(validate)

        self.model = None
    # end

    def load_data(self, path: str):
        data = []
        labels = set()
        label_num = {}

        # read CSV, add labels to label set
        with open(path, newline="") as f:
            file_contents = csv.reader(f, delimiter=",")
            next(file_contents)
            for row in file_contents:
                lbl = row[-1]
                if lbl.isdigit(): 
                    lbl = int(lbl)
                    row[-1] = lbl
                labels.add(lbl)
                data.append(row)
            # end
        # end

        # build label lookup for verification later
        for i, v in enumerate(labels):
            if isinstance(v, int):
                label_num[v] = v
                self.labels[v] = v
            else:
                label_num[v] = i
                self.labels[i] = v
        # end

        # convert data to numeric values
        dcopy = data.copy()
        data = []
        for row in dcopy:
            *x, y = row
            data.append([*map(float, x), label_num[y]])
        # end

        return data
    # end

    def datasets(self):
        part_labels = ["train", "test", "validate"]
        partitions = [self.train, self.test, self.validate]
        n = sum(len(p) for p in partitions)
        for i, p in enumerate(partitions):
            print(f"\nPartition {i}:")
            print(f"\tname: '{part_labels[i]}'")
            print(f"\tsize: {len(p)} / {n}")
            print(f"\tpcnt: {100*len(p)/n} %")
        # end
    # end

    def report(self, mod, X, Y):
        """Generate a classification report."""

        Y_true = Y
        Y_pred = mod.predict(X)

        report = classification_report(
            Y_true, Y_pred,
            # labels=[*map(str, self.labels.keys())],
            # target_names=[*self.labels.values()]
        )

        print(report)
    # end
# end


def process_data(X: list, n: int) -> list:
    """Scale and perform PCA on 'X'"""
    pca = PCA(n_components=n)
    X = preprocessing.scale(X)
    return pca.fit_transform(X)
# end


def compare_models(model: Model) -> dict:
    comps = [8, 16, 32, 64]
    results = {
        "support vector machine": [],
        "k-nearest neighbors": [],
        "logistic regression": [],
        "decision tree": []
    }
    classifiers = {
        "support vector machine": SVC(),
        "k-nearest neighbors": KNeighborsClassifier(
            weights="distance", metric="manhattan", n_jobs=-1
        ),
        "logistic regression": LogisticRegression(max_iter=1000),
        "decision tree": DecisionTreeClassifier()
    }
    for name, classifier in classifiers.items():
        for n in comps:
            xt = process_data(model.train.X, n)
            yt = model.train.Y

            xv = process_data(model.validate.X, n)
            yv = model.validate.Y

            classifier.fit(xt, yt)
            score = round(classifier.score(xv, yv), 5)

            res = f"PCA {n}: ".rjust(16) + f"{score}"
            results[name].append(res)
        # end
    # end
    return results
# end
