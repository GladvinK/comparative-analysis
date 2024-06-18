import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sentence_transformers import SentenceTransformer
from sklearn import linear_model
from sklearn.preprocessing import LabelBinarizer # Import LabelBinarizer
import matplotlib.pyplot as plt

# Get cleaned data set
df = pd.read_csv("../resource/newData.csv")

# Sentence encoding for model input
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Sentences are encoded by calling model.encode()
embedding = model.encode(df['text_'])

# Split data as test and train
x_train, x_test, y_train, y_test = train_test_split(embedding, df['label'], test_size=0.25)

# Random Forest model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_y_pred = rf.predict(x_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("Random Forest Accuracy:", rf_accuracy)

# Calculate accuracy for each class separately
class_accuracy = []
# Get unique class labels in y_test
unique_labels = y_test.unique()
for label in unique_labels:
 mask = y_test == label
 accuracy = accuracy_score(y_test[mask], rf_y_pred[mask])
 class_accuracy.append((label, accuracy))
# Display the results
for label, accuracy in class_accuracy:
 print(f"Class {label} - Accuracy: {accuracy}")

# Calculate precision, recall, and F1-score for each class separately
class_precision = precision_score(y_test, rf_y_pred, labels=['CG', 'OR'], average=None)
class_recall = recall_score(y_test, rf_y_pred, labels=['CG', 'OR'], average=None)
class_f1 = f1_score(y_test, rf_y_pred, labels=['CG', 'OR'], average=None)
# Display the results
for i, label in enumerate(['CG', 'OR']):
 print(f"Class {label} - Precision: {class_precision[i]}, Recall: {class_recall[i]}, F1-Score: {class_f1[i]}")


In [25]:
Logistic Regression Accuracy: 0.8914528616972826
C:\Users\GLADVIN\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:458: ConvergenceWarning: lbfgs f
ailed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
Increase the number of iterations (max_iter) or scale the data as shown in:
 https://scikit-learn.org/stable/modules/preprocessing.html (https://scikit-learn.org/stable/modules/preproc
essing.html)
Please also refer to the documentation for alternative solver options:
 https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression (https://scikit-learn.org/sta
ble/modules/linear_model.html#logistic-regression)
n_iter_i = _check_optimize_result(
# Transform categorical labels into binary labels
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
# Calculate ROC curve
rf_probabilities = rf.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_bin, rf_probabilities)
# Calculate ROC AUC
roc_auc = roc_auc_score(y_test_bin, rf_probabilities)
# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend(loc="lower right")
plt.show()


# Logistic Regression model
logr = linear_model.LogisticRegression()
logr.fit(x_train, y_train)
logr_y_pred = logr.predict(x_test)
logr_accuracy = accuracy_score(y_test, logr_y_pred)
print("Logistic Regression Accuracy:", logr_accuracy)

# Calculate accuracy for each class separately
class_accuracy = []
# Get unique class labels in y_test
unique_labels = y_test.unique()
for label in unique_labels:
 mask = y_test == label
 accuracy = accuracy_score(y_test[mask], logr_y_pred[mask])
 class_accuracy.append((label, accuracy))
# Display the results
for label, accuracy in class_accuracy:
 print(f"Class {label} - Accuracy: {accuracy}")

# Calculate precision, recall, and F1-score for each class separately
class_precision = precision_score(y_test, logr_y_pred, labels=['CG', 'OR'], average=None)
class_recall = recall_score(y_test, logr_y_pred, labels=['CG', 'OR'], average=None)
class_f1 = f1_score(y_test, logr_y_pred, labels=['CG', 'OR'], average=None)

# Display the results
for i, label in enumerate(['CG', 'OR']):
 print(f"Class {label} - Precision: {class_precision[i]}, Recall: {class_recall[i]}, F1-Score: {class_f1[i]}")

# Transform categorical labels into binary labels
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
# Calculate ROC curve
logr_probabilities = logr.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_bin, logr_probabilities)
# Calculate ROC AUC
roc_auc = roc_auc_score(y_test_bin, logr_probabilities)
# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.legend(loc="lower right")
plt.show()

from sklearn.tree import DecisionTreeClassifier

# Decision Tree model
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
dt_y_pred = dt.predict(x_test)
dt_accuracy = accuracy_score(y_test, dt_y_pred)
print("Decision Tree Accuracy:", dt_accuracy)


# Calculate accuracy for each class separately
class_accuracy = []
# Get unique class labels in y_test
unique_labels = y_test.unique()
for label in unique_labels:
 mask = y_test == label
 accuracy = accuracy_score(y_test[mask], dt_y_pred[mask])
 class_accuracy.append((label, accuracy))
# Display the results
for label, accuracy in class_accuracy:
 print(f"Class {label} - Accuracy: {accuracy}")

# Calculate precision, recall, and F1-score for each class separately
class_precision = precision_score(y_test, dt_y_pred, labels=['CG', 'OR'], average=None)
class_recall = recall_score(y_test, dt_y_pred, labels=['CG', 'OR'], average=None)
class_f1 = f1_score(y_test, dt_y_pred, labels=['CG', 'OR'], average=None)
# Display the results
for i, label in enumerate(['CG', 'OR']):
 print(f"Class {label} - Precision: {class_precision[i]}, Recall: {class_recall[i]}, F1-Score: {class_f1[i]}")


# Transform categorical labels into binary labels
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
# Calculate ROC curve
dt_probabilities = dt.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_bin, dt_probabilities)
# Calculate ROC AUC
roc_auc = roc_auc_score(y_test_bin, dt_probabilities)
# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Corrected line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree')
plt.legend(loc="lower right")
plt.show()

from sklearn.svm import SVC

# Support Vector Machine (SVM) model
svm = SVC(probability=True) # Set probability=True to enable ROC curve
svm.fit(x_train, y_train)
svm_y_pred = svm.predict(x_test)
svm_accuracy = accuracy_score(y_test, svm_y_pred)
print("SVM Accuracy:", svm_accuracy)


# Calculate accuracy for each class separately
class_accuracy = []
# Get unique class labels in y_test
unique_labels = y_test.unique()
for label in unique_labels:
 mask = y_test == label
 accuracy = accuracy_score(y_test[mask], svm_y_pred[mask])
 class_accuracy.append((label, accuracy))
# Display the results
for label, accuracy in class_accuracy:
 print(f"Class {label} - Accuracy: {accuracy}")

# Calculate precision, recall, and F1-score for each class separately
class_precision = precision_score(y_test, svm_y_pred, labels=['CG', 'OR'], average=None)
class_recall = recall_score(y_test, svm_y_pred, labels=['CG', 'OR'], average=None)
class_f1 = f1_score(y_test, svm_y_pred, labels=['CG', 'OR'], average=None)
# Display the results
for i, label in enumerate(['CG', 'OR']):
 print(f"Class {label} - Precision: {class_precision[i]}, Recall: {class_recall[i]}, F1-Score: {class_f1[i]}")

# Transform categorical labels into binary labels
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
# Calculate ROC curve
svm_probabilities = svm.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_bin, svm_probabilities)
# Calculate ROC AUC
roc_auc = roc_auc_score(y_test_bin, svm_probabilities)
# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Support Vector Machine (SVM)')
plt.legend(loc="lower right")
plt.show()


