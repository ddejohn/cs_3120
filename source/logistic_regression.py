import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics


class Model:
    def __init__(self):
        cols = [
            "pregnant", 
            "glucose", 
            "bp", 
            "skin", 
            "insulin", 
            "bmi", 
            "pedigree", 
            "age", 
            "outcome"
        ]

        self.features = [
            "glucose",
            "bp",
            "insulin",
            "bmi",
            "age"
        ]

        self.data = pd.read_csv(
            "../data/pima-indians-diabetes-database.csv",
            header=None,
            names=cols
        )

        tts = model_selection.train_test_split(
            self.data[self.features].to_numpy(),
            self.data.outcome,
            test_size=0.4
        )

        self.X_train, self.X_test, self.Y_train, self.Y_test = tts
        logreg = linear_model.LogisticRegression()
        logreg.fit(self.X_train, self.Y_train)

        self.Y_predict = logreg.predict(self.X_test)
        self.Y_proba = logreg.predict_proba(self.X_test)[::,1]
    # end


    def measures(self):
        accuracy = metrics.accuracy_score(self.Y_test, self.Y_predict)
        precision = metrics.precision_score(self.Y_test, self.Y_predict)
        recall = metrics.recall_score(self.Y_test, self.Y_predict)
        harmonic = 2*precision*recall/(precision + recall)

        print(metrics.confusion_matrix(self.Y_test, self.Y_predict))
        print(f"\nAccuracy:       {round(accuracy, 5)}")
        print(f"Precision:      {round(precision, 5)}")
        print(f"Recall:         {round(recall, 5)}")
        print(f"Harmonic mean:  {round(harmonic, 5)}\n")
    # end


    def roc_plot(self):
        fpr, tpr, _ = metrics.roc_curve(self.Y_test, self.Y_proba)
        auc = metrics.roc_auc_score(self.Y_test, self.Y_proba)

        fig = go.Figure(
            data=go.Scatter(
                x=fpr, y=tpr,
                line_width=5,
                line_color="lightsalmon"
            ),
            layout=go.Layout(
                height=950,
                width=950,
                title=f"<b>Area under ROC curve: {round(auc, 5)}</b>",
                plot_bgcolor="rgb(100,100,100)",
                xaxis_title="<b>False Positive Rate</b>",
                xaxis_gridcolor="rgb(120,120,120)",
                xaxis_zeroline=False,
                xaxis_range=[0.0, 1.0],
                yaxis_title="<b>True Positive Rate</b>",
                yaxis_gridcolor="rgb(120,120,120)",
                yaxis_zeroline=False,
                yaxis_range=[0.0, 1.0]
            )
        )

        fig.show()
    # end
# end
