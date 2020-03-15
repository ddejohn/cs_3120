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

        self.pima = pd.read_csv(
            "data/pima-indians-diabetes-database.csv",
            header=None,
            names=cols
        )

        tts = model_selection.train_test_split(
            self.pima[self.features].to_numpy(),
            self.pima.outcome,
            test_size=0.4
        )

        self.X_train, self.X_test, self.Y_train, self.Y_test = tts
        logreg = linear_model.LogisticRegression()
        logreg.fit(X_train, Y_train)

        self.Y_predict = logreg.predict(self.X_test)
        self.Y_proba = logreg.predict_proba(self.X_test)[::,1]
    # end


    def measures(self):
        accuracy = metrics.accuracy_score(self.Y_test, self.Y_predict)
        precision = metrics.precision_score(self.Y_test, self.Y_predict)
        recall = metrics.recall_score(self.Y_test, self.Y_predict)
        harmonic = 2*precision*recall/(precision + recall)

        print(metrics.confusion_matrix(self.Y_test, self.Y_predict))
        print(f"Accuracy:       {round(accuracy, 5)}")
        print(f"Precision:      {round(precision, 5)}")
        print(f"Recall:         {round(recall, 5)}")
        print(f"Harmonic mean:  {round(harmonic, 5)}\n")
    # end


    def roc_plot(self):
        fpr, tpr, _ = metrics.roc_curve(self.Y_test, self.Y_proba)
        auc = metrics.roc_auc_score(self.Y_test, self.Y_proba)

        fig = go.Figure(
            data=go.Scatter(x=fpr, y=tpr),
            layout=go.Layout(
                width=950, height=950,
                title=f"area under ROC curve: {round(auc, 5)}",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                legend_x=0, legend_y=1,
                legend_bgcolor="rgba(0,0,0,0.3)",
                legend_font_color="white"
            )
        )

        fig.show()
    # end
# end



with open("data/pima-indians-diabetes-database.csv") as f:
    pass