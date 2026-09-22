import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
class_names = data.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
pred = knn.predict(X_test_scaled)

print("KNN (k=5)")
print("Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=class_names))

print("\nAccuracy for different k values:")
for k in [1, 3, 5, 7, 9, 11]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    p = model.predict(X_test_scaled)
    print(f"k={k}: accuracy={accuracy_score(y_test, p):.4f}")

# Full metric breakdown (malignant = positive class)
cm = confusion_matrix(y_test, pred, labels=[0, 1])
TP, FN = cm[0, 0], cm[0, 1]
FP, TN = cm[1, 0], cm[1, 1]

accuracy = (TP + TN) / (TP + TN + FP + FN)
error_rate = 1 - accuracy
recall = TP / (TP + FN)
specificity = TN / (TN + FP)
precision = TP / (TP + FP)
f1 = 2 * precision * recall / (precision + recall)

prob_malignant = knn.predict_proba(X_test_scaled)[:, 0]
y_test_malignant = (y_test == 0).astype(int)
auc = roc_auc_score(y_test_malignant, prob_malignant)

print("\nFull metric breakdown (positive class = malignant)")
print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Error rate: {error_rate:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# k-fold cross-validation (5 folds), scaling done inside the pipeline to avoid data leakage
cv_pipeline = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))])
cv_scores = cross_val_score(cv_pipeline, X, y, cv=5, scoring="accuracy")
print("\n5-fold Cross-Validation")
print("Fold accuracies:", cv_scores)
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

sample = X_test.iloc[[0]]
sample_scaled = scaler.transform(sample)
pred_class = class_names[knn.predict(sample_scaled)[0]]
print(f"\nSample prediction -> KNN: {pred_class}")
