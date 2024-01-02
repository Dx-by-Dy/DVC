import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from dvclive import Live

df = pd.read_csv('iris_data.csv')
x_train, x_test, y_train, y_test = train_test_split(df.drop(["target"], axis=1), df["target"], test_size=0.5, random_state=0)

clf = MLPClassifier(hidden_layer_sizes=(50, 10), max_iter=1000, random_state=42)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

LC = clf.loss_curve_

data = np.column_stack((range(len(LC)), LC))

with Live(save_dvc_exp=True) as live:
    live.log_plot(Live, "MLP_50_10", data, "0", "1")