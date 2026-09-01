import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("11_mushroom_edibility.csv")
df = df.drop(columns=["SampleID"])

def entropy(labels):
    vals, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p))

target_entropy = entropy(df["Class"])
print(f"Entropy of Class: {target_entropy:.4f}")
print(df["Class"].value_counts())

encoders = {}
df_enc = df.copy()
for col in df_enc.columns:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    encoders[col] = le

X = df_enc.drop(columns=["Class"])
y = df_enc["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

print("\nNaive Bayes")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(confusion_matrix(y_test, nb_pred))
print(classification_report(y_test, nb_pred, target_names=encoders["Class"].classes_))

dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\nDecision Tree")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print(confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred, target_names=encoders["Class"].classes_))

sample = X_test.iloc[[0]]
nb_class = encoders["Class"].inverse_transform(nb.predict(sample))[0]
dt_class = encoders["Class"].inverse_transform(dt.predict(sample))[0]
print(f"\nSample prediction -> Naive Bayes: {nb_class}, Decision Tree: {dt_class}")
