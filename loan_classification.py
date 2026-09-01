import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("07_loan_approval.csv")
df = df.drop(columns=["ApplicantID"])

def entropy(labels):
    vals, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p))

target_entropy = entropy(df["LoanApproved"])
print(f"Entropy of LoanApproved: {target_entropy:.4f}")
print(df["LoanApproved"].value_counts())

encoders = {}
df_enc = df.copy()
categorical_cols = df_enc.select_dtypes(include=["object", "str"]).columns
for col in categorical_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    encoders[col] = le

X = df_enc.drop(columns=["LoanApproved"])
y = df_enc["LoanApproved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

print("\nNaive Bayes")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(confusion_matrix(y_test, nb_pred))
print(classification_report(y_test, nb_pred, target_names=encoders["LoanApproved"].classes_))

dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\nDecision Tree")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print(confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred, target_names=encoders["LoanApproved"].classes_))

sample = X_test.iloc[[0]]
nb_class = encoders["LoanApproved"].inverse_transform(nb.predict(sample))[0]
dt_class = encoders["LoanApproved"].inverse_transform(dt.predict(sample))[0]
print(f"\nSample prediction -> Naive Bayes: {nb_class}, Decision Tree: {dt_class}")
