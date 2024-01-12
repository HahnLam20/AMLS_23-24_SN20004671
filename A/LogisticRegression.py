import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from PIL import Image as im

#=======================sklearn imports=======================
from sklearn.utils import shuffle
from sklearn.model_selection import learning_curve, LearningCurveDisplay, cross_validate, GridSearchCV, validation_curve, ValidationCurveDisplay
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression

#Getting path for main directory for dataset and results
path = os.getcwd()
dataset_path = os.path.join(path, 'datasets')
result_path = os.path.join(path, 'A/results/KNN')

#Asserting relative imports from pipeline as needed
pipeline_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(pipeline_dir, '..'))
from pipeline.preprocessing import preprocess

#======================initialise dataset======================
pneumdata = np.load(os.path.join(dataset_path, 'pneumoniamnist.npz'))

#extracting images and labels from dataset
train_images = pneumdata['train_images']
val_images = pneumdata['val_images']
test_images = pneumdata['test_images']
train_labels = pneumdata['train_labels']
val_labels = pneumdata['val_labels']
test_labels = pneumdata['test_labels']

#======================preprocessing images======================
preprocess(train_images, train_labels)

#======================reshaping images=============================
train_images = train_images.reshape(len(train_images), -1)
test_images = test_images.reshape(len(test_images), -1)

#======================Training Logistic Regressor======================
model = LogisticRegression(random_state=0, max_iter=200, solver = 'liblinear') #basic case
model_l1 = LogisticRegression(penalty = 'l1', random_state=0, max_iter=200, solver = 'liblinear') #LASSO regularization
model.fit(train_images, train_labels)
model_l1.fit(train_images, train_labels)

#======================Cross Validation======================
cv_results = cross_validate(model, train_images, train_labels, cv=5)
cv_results_l1 = cross_validate(model_l1, train_images, train_labels, cv=5)
print("5-fold cross-validation results (without regularisation): {}".format(cv_results['test_score']))
print("5-fold cross-validation results (with LASSO regularisation): {}".format(cv_results_l1['test_score']))
print("Mean accuracy with cross validation (without regularisation): {}".format(cv_results['test_score'].mean()))
print("Mean accuracy with cross validation (with LASSO regularisation): {}".format(cv_results_l1['test_score'].mean()))

#Grid Search for hyperparameter tuning
param_grid = {'C': np.arange(0.1, 1.1, 0.1)}
grid = GridSearchCV(model_l1, param_grid, cv=5)
grid.fit(train_images, train_labels)
print("Best value of C: ", grid.best_params_['C'])
print("Highest achieved accuracy: {}".format(grid.best_score_))

#======================Plotting Validation Curve for Regularisation======================
param_range = np.arange(0.1, 1.1, 0.1)
train_scores, test_scores = validation_curve(model_l1, train_images, train_labels, param_name="C", param_range=param_range, cv=5)
display = ValidationCurveDisplay(
    param_name="C",
    param_range=param_range,
    train_scores=train_scores,
    test_scores=test_scores,
)
display.plot()
plt.savefig(os.path.join(result_path, 'LR/LRValidationCurve.png'))

#======================Predicting Labels======================
pred_labels = model.predict(test_images)
pred_labels_l1 = model_l1.predict(test_images)
print("Accuracy score without regularisation: {}".format(accuracy_score(test_labels, pred_labels)))
print("Accuracy score with LASSO regularisation: {}".format(accuracy_score(test_labels, pred_labels_l1)))

#======================Confusion Matrix======================
cm = confusion_matrix(test_labels, pred_labels)
cm_l1 = confusion_matrix(test_labels, pred_labels_l1)
print("Confusion Matrix without regularisation: \n{}".format(cm))
print("Confusion Matrix with LASSO regularisation: \n{}".format(cm_l1))

#======================Classification Report======================
print("Classification Report without regularisation: \n{}".format(classification_report(test_labels, pred_labels)))
print("Classification Report with LASSO regularisation: \n{}".format(classification_report(test_labels, pred_labels_l1)))

#==================Calculation of Sensitivity and Specificity==================
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) * 100
specificity = tn / (tn + fp) * 100
print(f'Sensitivity without regularisation: {sensitivity}%')
print(f'Specificity without regularisation: {specificity}%')

tn, fp, fn, tp = cm_l1.ravel()
sensitivity = tp / (tp + fn) * 100
specificity = tn / (tn + fp) * 100
print(f'Sensitivity with LASSO regularisation: {sensitivity}%')
print(f'Specificity with LASSO regularisation: {specificity}%')

#======================Plotting Confusion Matrix======================
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix without regularisation')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(result_path, 'LR/LRConfusionMatrix.png'))

disp = ConfusionMatrixDisplay(confusion_matrix=cm_l1)
disp.plot()
plt.title('Confusion Matrix with LASSO regularisation')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(result_path, 'LR/LRConfusionMatrixL1.png'))



#===============Plotting Learning Curves================
#Learning curve without regularisation
common_params = {
    'cv': 5,
    'n_jobs': -1,
    'train_sizes': np.linspace(0.1, 1.0, 10),
    'shuffle': True,
    'random_state': 0,
    "scoring": "accuracy"
}

train_sizes, train_scores, test_scores = learning_curve(model, train_images, train_labels, **common_params)
display = LearningCurveDisplay(
    train_sizes=train_sizes,
    train_scores=train_scores,
    test_scores=test_scores,
    score_name = 'Score'
)
display.plot()
plt.savefig(os.path.join(result_path, 'LR/LRLearningCurve.png'))

#Learning curve with LASSO regularisation
train_sizes, train_scores, test_scores = learning_curve(model_l1, train_images, train_labels, **common_params)
display = LearningCurveDisplay(
    train_sizes=train_sizes,
    train_scores=train_scores,
    test_scores=test_scores,
    score_name = 'Score'
)
display.plot()
plt.savefig(os.path.join(result_path, 'LR/LRLearningCurveL1.png'))
