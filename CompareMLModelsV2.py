'''
Ryan Pierce
ID: 2317826
EECS 690 - Intro to Machine Learning, Python Project 3
'''

print("\nPROGRAM STARTING\n")

# Load libraries
import numpy as np
from sys import stdout
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

print ('Libraries loaded successfully!')

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
print ('Dataset loaded successfully!\n')

# Array containing initial dataset
data_array = dataset.values

# For X: select all rows, but only columns indexed from 0 to 3 (inputs)
X = data_array[:, 0:4]

# For y: select all rows, but only the last column (outputs)
y = data_array[:, 4]

# Splitting the data in half, then creating two folds.
X_train_fold1, X_test_fold1, y_train_fold1, y_test_fold1 = train_test_split(X, y, test_size = 0.50, random_state = 1)

X_train_fold2 = X_test_fold1
X_test_fold2 = X_train_fold1
y_train_fold2 = y_test_fold1
y_test_fold2 = y_train_fold1


# START Gaussian Naive-Bayesian Model
model = GaussianNB()

# Fit the model to each fold, creating two different sets of predictions
model.fit(X_train_fold1, y_train_fold1)
predictions_fold1 = model.predict(X_test_fold1)
model.fit(X_train_fold2, y_train_fold2)
predictions_fold2 = model.predict(X_test_fold2)

# Concatenate the prediction sets from both folds
predictions = np.concatenate((predictions_fold1, predictions_fold2))
y_test = np.concatenate((y_test_fold1, y_test_fold2))

print('|----- GaussianNB Model -----|\n')
# Accuracy of predictions
print('Accuracy:')
print(accuracy_score(y_test, predictions))
print('\n')
# Confusion Matrix for predictions
print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))
print('\n')
print('|----------------------------|\n')
# END Gaussian Naive


# START Linear Regression Model
model = LinearRegression()

# Linear Regression Model requires outputs to be integers, so I am
# using a 'for' loop to change the values of classes to ints in the
# array
for i in range(len(y_train_fold1)):
    if y_train_fold1[i] == 'Iris-setosa':
        y_train_fold1[i] = 1
    elif y_train_fold1[i] == 'Iris-virginica':
        y_train_fold1[i] = 2
    elif y_train_fold1[i] == 'Iris-versicolor':
        y_train_fold1[i] = 3

for i in range(len(y_train_fold2)):
    if y_train_fold2[i] == 'Iris-setosa':
        y_train_fold2[i] = 1
    elif y_train_fold2[i] == 'Iris-virginica':
        y_train_fold2[i] = 2
    elif y_train_fold2[i] == 'Iris-versicolor':
        y_train_fold2[i] = 3

model.fit(X_train_fold1, y_train_fold1)
predictions_fold1 = model.predict(X_test_fold1)
model.fit(X_train_fold2, y_train_fold2)
predictions_fold2 = model.predict(X_test_fold2)
predictions = []
y_test = []

# Round to nearest int for all floats in predictions and y_test
for i in np.concatenate((predictions_fold1, predictions_fold2)):
    if i >= 2.5:
        predictions.append(3)
    elif i >= 1.5:
        predictions.append(2)
    else:
        predictions.append(1)
for i in np.concatenate((y_test_fold1, y_test_fold2)):
    if i >= 2.5:
        y_test.append(3)
    elif i >= 1.5:
        y_test.append(2)
    else:
        y_test.append(1)

print('|----- Linear Regression Model -----|\n')
print('Accuracy:')
print(accuracy_score(y_test, predictions))
print('\n')
print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))
print('\n')
print('|-----------------------------------|\n')
# END Linear Regression Model


# START Polynomial of Degree 2 Regression
model = make_pipeline(PolynomialFeatures(2), Ridge())

model.fit(X_train_fold1, y_train_fold1)
predictions_fold1 = model.predict(X_test_fold1)
model.fit(X_train_fold2, y_train_fold2)
predictions_fold2 = model.predict(X_test_fold2)

predictions = []
for i in np.concatenate([predictions_fold1, predictions_fold2]):
    if i >= 2.5:
        predictions.append(3)
    elif i >= 1.5:
        predictions.append(2)
    else:
        predictions.append(1)

print('|----- Polynomial of Degree 2 Regression Model -----|\n')
print('Accuracy:')
print(accuracy_score(y_test, predictions))
print('\n')
print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))
print('\n')
print('|---------------------------------------------------|\n')
# END Polynomial of Degree 2 Regression


# START Polynomial of Degree 3 Regression
model = make_pipeline(PolynomialFeatures(3), Ridge())

model.fit(X_train_fold1, y_train_fold1)
predictions_fold1 = model.predict(X_test_fold1)
model.fit(X_train_fold2, y_train_fold2)
predictions_fold2 = model.predict(X_test_fold2)

predictions = []
for i in np.concatenate([predictions_fold1, predictions_fold2]):
    if i >= 2.5:
        predictions.append(3)
    elif i >= 1.5:
        predictions.append(2)
    else:
        predictions.append(1)

print('|----- Polynomial of Degree 3 Regression Model -----|\n')
print('Accuracy:')
print(accuracy_score(y_test, predictions))
print('\n')
print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))
print('\n')
print('|---------------------------------------------------|\n')
# END Polynomial of Degree 3 Regression

X_train_fold1, X_test_fold1, y_train_fold1, y_test_fold1 = train_test_split(X, y, test_size = 0.50, random_state = 1)

X_train_fold2 = X_test_fold1
X_test_fold2 = X_train_fold1
y_train_fold2 = y_test_fold1
y_test_fold2 = y_train_fold1

# START k-Closest Neighbors (kNN)
model = KNeighborsClassifier()
model.fit(X_train_fold1, y_train_fold1)
predictions_fold1 = model.predict(X_test_fold1)
model.fit(X_train_fold2, y_train_fold2)
predictions_fold2 = model.predict(X_test_fold2)

predictions = np.concatenate([predictions_fold1, predictions_fold2])
y_test = np.concatenate((y_test_fold1, y_test_fold2))

print('|----- k-Closest Neighbors Model -----|\n')
print('Accuracy:')
print(accuracy_score(y_test, predictions))
print('\n')
print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))
print('\n')
print('|-------------------------------------|\n')
#END k-Closest Neighbors


# START Linear Discriminant Analysis (LDA)
model = LinearDiscriminantAnalysis()
model.fit(X_train_fold1, y_train_fold1)
predictions_fold1 = model.predict(X_test_fold1)
model.fit(X_train_fold2, y_train_fold2)
predictions_fold2 = model.predict(X_test_fold2)

predictions = np.concatenate([predictions_fold1, predictions_fold2])
y_test = np.concatenate((y_test_fold1, y_test_fold2))

print('|----- LDA Model -----|\n')
print('Accuracy:')
print(accuracy_score(y_test, predictions))
print('\n')
print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))
print('\n')
print('|---------------------|\n')
# END LDA


# START Neural Network: MLPClassification
model = MLPClassifier(solver='lbfgs', alpha=1, hidden_layer_sizes=(5, 2), random_state=1, max_iter=500)
model.fit(X_train_fold1, y_train_fold1)
predictions_fold1 = model.predict(X_test_fold1)
model.fit(X_train_fold2, y_train_fold2)
predictions_fold2 = model.predict(X_test_fold2)

predictions = np.concatenate([predictions_fold1, predictions_fold2])
y_test = np.concatenate((y_test_fold1, y_test_fold2))

print('|----- NN MLPClassification -----|\n')
print('Accuracy:')
print(accuracy_score(y_test, predictions))
print('\n')
print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))
print('\n')
print('|--------------------------------|\n')
# END Neural Network


# START SVM: SVC
model = SVC(gamma='auto')
model.fit(X_train_fold1, y_train_fold1)
predictions_fold1 = model.predict(X_test_fold1)
model.fit(X_train_fold2, y_train_fold2)
predictions_fold2 = model.predict(X_test_fold2)

predictions = np.concatenate([predictions_fold1, predictions_fold2])
y_test = np.concatenate((y_test_fold1, y_test_fold2))

print('|----- SVM SVC -----|\n')
print('Accuracy:')
print(accuracy_score(y_test, predictions))
print('\n')
print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))
print('\n')
print('|-------------------------|\n')
# END SVM: LinearSVC