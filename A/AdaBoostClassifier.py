from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import scienceplots
import configparser
from PIL import Image as im

#sklearn imports
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, GridSearchCV, learning_curve, LearningCurveDisplay, validation_curve, ValidationCurveDisplay

#torch imports
import torch
from torchvision import datasets
from torchvision.transforms import v2, transforms

#Plotting Style
plt.style.use(['science', 'ieee'])  #Needs LaTeX installed. If not, comment out and use plt.style.use(['science', 'no-latex']) instead.

#getting path for main directory for dataset and results
path = os.getcwd()
dataset_path = os.path.join(path, 'datasets')
result_path = os.path.join(path, 'A/results/AdaBoostDT')

#Asserting relative imports from pipeline as needed
pipeline_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(pipeline_dir, '..'))
from pipeline.preprocessing import preprocess

#initialise dataset for visualisqation
pneumdata = np.load(os.path.join(dataset_path, 'pneumoniamnist.npz'))

#extracting images and labels from dataset
train_images = pneumdata['train_images']
val_images = pneumdata['val_images']
test_images = pneumdata['test_images']
train_labels = pneumdata['train_labels']
val_labels = pneumdata['val_labels']
test_labels = pneumdata['test_labels']

#changing image data shape for adaboost
#preprocessing using function
preprocess(train_images, train_labels)

#reshaping images
train_images = train_images.reshape(len(train_images), 28*28)
val_images = val_images.reshape(len(val_images), 28*28)
test_images = test_images.reshape(len(test_images), 28*28)

#initialise adaboost classifier and train
#use hyperparameters from command line
config = configparser.ConfigParser()
config.read('config.ini')
try:
    n_estimators = int(config['USER']['n_estimators'])
except(KeyError):
    n_estimators = int(config['DEFAULT']['n_estimators'])
try:
    learning_rate = float(config['USER']['learning_rate'])
except(KeyError):
    learning_rate = float(config['DEFAULT']['learning_rate'])
try:
    algorithm = config['USER']['algorithm']
except(KeyError):
    algorithm = config['DEFAULT']['algorithm']
print("Using: n_estimators = {}, learning_rate = {}, algorithm = {}".format(n_estimators, learning_rate, algorithm))
weak_learner = DecisionTreeClassifier(random_state = 0)
model = AdaBoostClassifier(estimator=weak_learner,
                           n_estimators=n_estimators,
                           learning_rate=learning_rate,
                           algorithm=algorithm,
                           random_state=None)
weak_learner.fit(train_images, train_labels)
model.fit(train_images, train_labels)

#grid search for best params
param_grid = {
    'n_estimators': [10, 20, 50, 100, 200],
    'learning_rate': [0.1, 0.5, 1.0, 2.0],
    'algorithm': ['SAMME', 'SAMME.R']
}
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(train_images, train_labels)
print("Best Number of Estimators: {}".format(grid.best_params_['n_estimators']))
print("Best Learning Rate: {}".format(grid.best_params_['learning_rate']))
print("Best Algorithm: {}".format(grid.best_params_['algorithm']))
print("Highest achieved accuracy: {}".format(grid.best_score_))

#plotting validation curves
#number of estimators
param_range = np.arange(1, 101, 10)
train_scores, test_scores = validation_curve(model, train_images, train_labels, param_name="n_estimators", param_range=param_range, cv=5)
display = ValidationCurveDisplay(
    param_name="n_estimators",
    param_range=param_range,
    train_scores=train_scores,
    test_scores=test_scores,
    score_name="Score"
)
display.plot()
plt.savefig(os.path.join(result_path, 'adaboost_n_estimators_validation_curve.png'))

#using and training optimised model
optimised_model = AdaBoostClassifier(estimator=weak_learner,
                            n_estimators=grid.best_params_['n_estimators'],
                            learning_rate=grid.best_params_['learning_rate'],
                            algorithm=grid.best_params_['algorithm'],
                            random_state=None)
optimised_model.fit(train_images, train_labels)


#misclassification error function
def misclassification_error(y_true, y_pred):
    return np.sum(y_true != y_pred) / len(y_true)

#Display misclassification error and accuracy and compare vs weak learner
weak_learner_misclassification_error = misclassification_error(val_labels, weak_learner.predict(val_images))
adaboost_misclassification_error = misclassification_error(val_labels, optimised_model.predict(val_images))
print(f'Weak learner misclassification error: {weak_learner_misclassification_error}')
print(f'AdaBoost misclassification error: {adaboost_misclassification_error}')
print(f'Weak learner accuracy: {weak_learner.score(val_images, val_labels)}')
print(f'AdaBoost accuracy: {model.score(val_images, val_labels)}')

#compare boosting vs weak learner
boosting_acc = pd.DataFrame({
    'Weak Learner': [weak_learner.score(val_images, val_labels)],
    'AdaBoost': [optimised_model.score(val_images, val_labels)]
})
boosting_acc.plot(kind='bar', title='Boosting vs Weak Learner Accuracy')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(result_path, 'boosting_vs_weak_learner_accuracy.png'))

#making predictions on test set using adaboost and weak learner
weak_learner_pred = weak_learner.predict(test_images)
print(f'Weak learner accuracy: {accuracy_score(test_labels, weak_learner_pred)}')
print(f'Weak learner confusion matrix: {confusion_matrix(test_labels, weak_learner_pred)}')
adaboost_pred = optimised_model.predict(test_images)
print(f'Accuracy: {accuracy_score(test_labels, adaboost_pred)}')
print(f'Confusion Matrix: {confusion_matrix(test_labels, adaboost_pred)}')

#plotting confusion matrix with values
plt.figure(figsize=(10,10))
cm = confusion_matrix(test_labels, adaboost_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(result_path, 'confusion_matrix.png'))

#calculation of sensitivity and specificity
tn, fp, fn, tp = confusion_matrix(test_labels, adaboost_pred).ravel()
sensitivity = tp / (tp + fn) * 100
specificity = tn / (tn + fp) * 100
print(f'Sensitivity: {sensitivity}%')
print(f'Specificity: {specificity}%')

#Classification Report
print(classification_report(test_labels, adaboost_pred))
print(classification_report(test_labels, weak_learner_pred))

#plotting sample predictions
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(adaboost_pred[i], test_labels[i]))
plt.savefig(os.path.join(result_path, 'adaboost_predictions.png'))

#plotting learning curves
#learning curve (weak learner)
train_sizes, train_scores, test_scores = learning_curve(weak_learner, train_images, train_labels, cv=5)
display = LearningCurveDisplay(
    train_sizes=train_sizes,
    train_scores=train_scores,
    test_scores=test_scores,
    score_name = 'Score'
)
display.plot()
plt.savefig(os.path.join(result_path, 'weak_learner_learning_curve.png'))

#learning curve (Adaboost)
train_sizes, train_scores, test_scores = learning_curve(optimised_model, train_images, train_labels, cv=5)
display = LearningCurveDisplay(
    train_sizes=train_sizes,
    train_scores=train_scores,
    test_scores=test_scores,
    score_name = 'Score'
)
display.plot()
plt.savefig(os.path.join(result_path, 'adaboost_learning_curve.png'))
