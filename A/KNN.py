#general imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import configparser
import scienceplots
from PIL import Image as im

#sklearn imports
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_validate, GridSearchCV, learning_curve, LearningCurveDisplay, validation_curve, ValidationCurveDisplay
from sklearn.neighbors import KNeighborsClassifier

#torch imports
from torchvision.transforms import v2, transforms

#Plotting Style
plt.style.use(['science', 'ieee'])  #Needs LaTeX installed. If not, comment out and use plt.style.use(['science', 'no-latex']) instead.

#Getting path for main directory for dataset and results
path = os.getcwd()
dataset_path = os.path.join(path, 'datasets')
result_path = os.path.join(path, 'A/results/KNN')

#Asserting relative imports from pipeline as needed
pipeline_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(pipeline_dir, '..'))
from pipeline.preprocessing import preprocess

#initialise dataset
pneumdata = np.load(os.path.join(dataset_path, 'pneumoniamnist.npz'))

#extracting images and labels from dataset
train_images = pneumdata['train_images']
val_images = pneumdata['val_images']
test_images = pneumdata['test_images']
train_labels = pneumdata['train_labels']
val_labels = pneumdata['val_labels']
test_labels = pneumdata['test_labels']

#preprocessing images
preprocess(train_images, train_labels)

#reshaping images for KNN
train_images_reshaped = train_images.reshape(train_images.shape[0], -1)
test_images_reshaped = test_images.reshape(test_images.shape[0], -1)

#training model - determine n_neighbors from user input
config = configparser.ConfigParser()
config.read('config.ini')
try:
    k_neighbors = int(config['USER']['k_neighbors'])
except(KeyError):
    k_neighbors = int(config['DEFAULT']['k_neighbors'])
print("Using: k_neighbors = {}".format(k_neighbors))
knn = KNeighborsClassifier(n_neighbors=k_neighbors)
knn.fit(train_images_reshaped, train_labels)

#cross-validation for hyperparameter tuning
cv_results = cross_validate(knn, train_images_reshaped, train_labels, cv=5)
print("5-fold cross-validation results: {}".format(cv_results['test_score']))

#plotting validation curve
param_range = np.arange(1, 11)
train_scores, test_scores = validation_curve(knn, train_images_reshaped, train_labels, param_name="n_neighbors", param_range=param_range, cv=5)
display = ValidationCurveDisplay(
    param_name="n_neighbors",
    param_range=param_range,
    train_scores=train_scores,
    test_scores=test_scores,
    score_name="Score"
)
display.plot()
plt.savefig(os.path.join(result_path, 'KNN_validation_curve.png'))

#grid search cross validation
param_grid = {'n_neighbors': np.arange(1, 11)}
grid = GridSearchCV(knn, param_grid, cv=5)
grid.fit(train_images_reshaped, train_labels)
print("Best Number of Neighbours: {}".format(grid.best_params_['n_neighbors']))
print("Highest achieved accuracy: {}".format(grid.best_score_))

#retraining new model with best params
optimised_knn = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'])
optimised_knn.fit(train_images_reshaped, train_labels)

#predicting labels
pred_labels = optimised_knn.predict(test_images_reshaped)

#evaluating model
print(confusion_matrix(test_labels, pred_labels))
print(accuracy_score(test_labels, pred_labels))

#plotting results
#Plotting predictions
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(test_images[i], cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(pred_labels[i], test_labels[i]))

#Printing Classification Report
print(classification_report(test_labels, pred_labels))

#plotting confusion matrix
plt.figure(figsize=(10,10))
cm = confusion_matrix(test_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(result_path, 'CNN3_confusion_matrix.png'))

#calculation of sensitivity and specificity
tn, fp, fn, tp = confusion_matrix(test_labels, pred_labels).ravel()
sensitivity = tp / (tp + fn) * 100
specificity = tn / (tn + fp) * 100
print(f'Sensitivity: {sensitivity}%')
print(f'Specificity: {specificity}%')


#plotting learning curves
train_sizes, train_scores, test_scores = learning_curve(optimised_knn, train_images_reshaped, train_labels, cv=5)
display = LearningCurveDisplay(
    train_sizes=train_sizes,
    train_scores=train_scores,
    test_scores=test_scores,
    score_name = 'Score'
)
display.plot()
plt.title("KNN Learning Curve")
plt.savefig(os.path.join(result_path, 'KNN_Learning_Curve.png'))
