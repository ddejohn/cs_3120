import csv
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


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
        # random.seed(42)
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

        # self.datasets()
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
            x = [*map(float, x)]
            y = label_num[y]
            data.append([*x, y])
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

    def report(self, mod, dataset: Data):
        """Generate a classification report."""

        Y_true = dataset.Y
        Y_pred = mod.predict(dataset.X)

        report = classification_report(
            Y_true, Y_pred,
            labels=[*self.labels.keys()],
            target_names=[*self.labels.values()]
        )

        print(report)
    # end
# end
