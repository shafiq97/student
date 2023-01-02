import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv("student-mat.csv")

# 3
data['pass'] = data.apply(lambda row: 1 if row['G1'] + row['G2'] + row['G3'] >= 35 else 0, axis=1)
data.to_csv('student-mat.csv"', index=False)

# 2
data = pd.get_dummies(data, columns=['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'])

# 4.1
# data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)

# 4.2
# Initialize the StandardScaler object
scaler = StandardScaler()
# Fit the StandardScaler object to the training data
xtrain_scaled = scaler.fit_transform(xtrain)
# Transform the test data using the fitted StandardScaler object
X_test_scaled = scaler.transform(xtrain)


# 4.3
svm_linear_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear"))
])

svm_linear_pipeline.fit(xtrain, ytrain)

y_pred = svm_linear_pipeline.predict(xtest)

accuracy = accuracy_score(ytest, y_pred)
precision = precision_score(ytest, y_pred, average='micro')
recall = recall_score(ytest, y_pred, average='micro')
f1 = f1_score(ytest, y_pred, average='micro')

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1-Score:", f1)

# confusion_matrix = confusion_matrix(ytest, y_pred)
# display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
# display.plot()

# 4.4
# Build and fit the non-linear SVM model
linear_svm = SVC(kernel="linear")
linear_svm.fit(xtrain, ytrain)
y_pred_linear = linear_svm.predict(xtest)
accuracy_linear = accuracy_score(ytest, y_pred_linear)
precision_linear = precision_score(ytest, y_pred_linear, average='micro')
recall_linear = recall_score(ytest, y_pred_linear, average='micro')
f1_linear = f1_score(ytest, y_pred_linear, average='micro')

# Print results for linear SVM classifier
print("Linear SVM Classifier:")
print("Accuracy:", accuracy_linear)
print("Precision:", precision_linear)
print("Recall:", recall_linear)
print("F1-Score:", f1_linear)

nonlinear_svm = SVC(kernel="rbf")
nonlinear_svm.fit(xtrain, ytrain)
# Predict the test set using the non-linear SVM model
y_pred = nonlinear_svm.predict(xtest)
# Calculate and print the classification performance metrics for the non-linear SVM model
accuracy = accuracy_score(ytest, y_pred)
precision = precision_score(ytest, y_pred, average='micro')
recall = recall_score(ytest, y_pred, average='micro')
f1 = f1_score(ytest, y_pred, average='micro')
print("Non Linear SVM Classifier:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# 4.4.1
parameters = {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]}
model = SVC()
scoring = 'accuracy'
grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring=scoring, cv=5)
grid_search.fit(xtrain, ytrain)
optimal_params = grid_search.best_params_
optimal_model = SVC(kernel=optimal_params['kernel'], C=optimal_params['C'])
optimal_model.fit(xtrain, ytrain)
y_pred = optimal_model.predict(xtest)

accuracy = accuracy_score(ytest, y_pred)
precision = precision_score(ytest, y_pred, average='micro')
recall = recall_score(ytest, y_pred, average='micro')
f1 = f1_score(ytest, y_pred, average='micro')

print(f'Accuracy: {accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1-Score: {f1:.3f}')
my_confusion_matrix = confusion_matrix(ytest, y_pred)

display = ConfusionMatrixDisplay(confusion_matrix=my_confusion_matrix)
display.plot()



