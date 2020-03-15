import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics

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

pima = pd.read_csv(
    "datasets/pima-indians-diabetes-database.csv",
    header=None,
    names=cols
)

features = ["glucose", "bp", "insulin", "bmi", "age"]

tts = model_selection.train_test_split(
    pima[features].to_numpy(),
    pima.outcome,
    test_size=0.4
)

X_train, X_test, Y_train, Y_test = tts
logreg = linear_model.LogisticRegression()
logreg.fit(X_train, Y_train)

Y_predict = logreg.predict(X_test)
Y_proba = logreg.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(Y_test, Y_proba)
accuracy = metrics.accuracy_score(Y_test, Y_predict)
precision = metrics.precision_score(Y_test, Y_predict)
recall = metrics.recall_score(Y_test, Y_predict)
harmonic = 2*precision*recall/(precision + recall)
roc = go.Scatter(x=fpr, y=tpr)
auc = metrics.roc_auc_score(Y_test, Y_proba)

print(pima)
print(pima[features])
print(metrics.confusion_matrix(Y_test, Y_predict))
print(f"Accuracy:       {accuracy}")
print(f"Precision:      {precision}")
print(f"Recall:         {recall}")
print(f"Harmonic mean:  {harmonic}\n")

fig = go.Figure(
    data=roc,
    layout=go.Layout(
        width=950, height=950,
        title=f"area under ROC curve: {auc}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend_x=0, legend_y=1,
        legend_bgcolor="rgba(0,0,0,0.3)",
        legend_font_color="white"
    )
)
fig.show()