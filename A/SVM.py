from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import configparser
import scienceplots
import json
from PIL import Image as im

#sklearn imports
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve, LearningCurveDisplay, validation_curve, ValidationCurveDisplay, cross_validate, GridSearchCV


#torch imports
from torchvision.transforms import v2, transforms

#Plotting Style
plt.style.use(['science', 'no-latex'])
#if no ieee style, comment out above line and uncomment below line
#plt.style.use(['science','no-latex'])

#Getting path for main directory for dataset and results
path = os.getcwd()
dataset_path = os.path.join(path, 'datasets')
result_path = os.path.join(path, 'A/results/SVM')

#Asserting relative imports from pipeline as needed
pipeline_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(pipeline_dir, '..'))
from pipeline.preprocessing import preprocess
from pipeline.resultsanalysis import gen_loss_plot, gen_acc_plot, gen_confusion_matrix, gen_json
from pipeline.utils import clean_folder
clean_folder(result_path)

#initialise dataset
pneumdata = np.load(os.path.join(dataset_path, 'pneumoniamnist.npz'))

#extracting images and labels from dataset
train_images = pneumdata['train_images']
val_images = pneumdata['val_images']
test_images = pneumdata['test_images']
train_labels = pneumdata['train_labels']
val_labels = pneumdata['val_labels']
test_labels = pneumdata['test_labels']

#preprocessing images and labels with image augmentation
# preprocess(train_images, train_labels)

#Implementing Linear and RBF SVM with hyperparameters
config = configparser.ConfigParser()
config.read('config.ini')
try:
    C = float(config['USER']['c_value'])
    gamma = float(config['USER']['gamma_value'])
except(KeyError):
    C = float(config['DEFAULT']['c_value'])
    gamma = float(config['DEFAULT']['gamma_value'])
print("Using: c = {}, gamma = {}".format(C, gamma))
linear_model = svm.SVC(kernel='linear', C = C)
rbf_model = svm.SVC(kernel='rbf', C = C, gamma=gamma)

#reshaping images for svm
train_images = train_images.reshape(len(train_images), 28*28)
val_images = val_images.reshape(len(val_images), 28*28)
test_images = test_images.reshape(len(test_images), 28*28)

#train svm
linear_model.fit(train_images, train_labels)
rbf_model.fit(train_images, train_labels)

#cross-validation for hyperparameter tuning
#linear case
cv_results = cross_validate(linear_model, train_images, train_labels, cv=5)
print(cv_results)
#rbf case
cv_results_rbf = cross_validate(rbf_model, train_images, train_labels, cv=5)
print(cv_results_rbf)

#Grid Search for Best Hyperparameters:
# #Linear SVM
# param_grid = {'C': [0.1, 1, 10, 100, 1000]}
# grid = GridSearchCV(linear_model, param_grid, refit=True, verbose=3)
# grid.fit(train_images, train_labels)

# #RBF SVM
# # param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.01, 0.0001, 0.000001, 0.00000001]}
# rbf_grid = GridSearchCV(rbf_model, param_grid, refit=True, verbose=3)
# rbf_grid.fit(train_images, train_labels)

#printing results together so it doesn't get missed in the output
# print("Best value of C (Linear): {}".format(grid.best_params_['C']))
# print("Highest achieved accuracy (Linear): {}".format(grid.best_score_))
# print("Best value of C (RBF): {}".format(rbf_grid.best_params_['C']))
# print("Best value of gamma (RBF): {}".format(rbf_grid.best_params_['gamma']))
# print("Highest achieved accuracy (RBF): {}".format(rbf_grid.best_score_))

#fitting optimum models
linear_model = svm.SVC(kernel='linear', C = 0.1)
rbf_model = svm.SVC(kernel='rbf', C = 10, gamma=1e-06)
linear_model.fit(train_images, train_labels)
rbf_model.fit(train_images, train_labels)

#Plotting Validation Curves
#Linear SVM Validation
C_range = np.logspace(-1, 3, 5)
train_scores, test_scores = validation_curve(linear_model, train_images, train_labels, param_name="gamma", param_range=C_range, cv=5)
display = ValidationCurveDisplay(
    param_name="C",
    param_range=C_range,
    train_scores=train_scores,
    test_scores=test_scores,
    score_name="Score"
)
display.plot()
plt.savefig(os.path.join(result_path, 'Linear_SVM_validation_curve.png'))

#RBF Validation
gamma_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(rbf_model, train_images, train_labels, param_name="gamma", param_range=gamma_range, cv=5)
display = ValidationCurveDisplay(
    param_name="gamma",
    param_range=gamma_range,
    train_scores=train_scores,
    test_scores=test_scores,
    score_name="Score"
)
display.plot()
plt.savefig(os.path.join(result_path, 'RBF_SVM_validation_curve.png'))

#Checking Misclassification Error on Validation Set
def misclassification_error(model, images, labels):
    '''
    Simple function to calculate the misclassification error of a model as a function
    of the accuracy score.
    :param model: input model
    :param images: images to predict
    :param labels: assigned labels of images
    :return:
    '''
    pred = model.predict(images)
    return 1 - accuracy_score(labels, pred)
print(f'Linear SVM misclassification error: {misclassification_error(linear_model, val_images, val_labels)}')
print(f'RBF SVM misclassification error: {misclassification_error(rbf_model, val_images, val_labels)}')

#Checking and Comparing Accuracy
linear_pred = linear_model.predict(test_images)
rbf_pred = rbf_model.predict(test_images)
print(f'Linear SVM accuracy: {accuracy_score(test_labels, linear_pred)}')
print(f'RBF SVM accuracy: {accuracy_score(test_labels, rbf_pred)}')

#Calculating Sensitivity and Specificity
#Linear SVM
tn, fp, fn, tp = confusion_matrix(test_labels, linear_pred).ravel()
lin_sensitivity = tp / (tp + fn) * 100
lin_specificity = tn / (tn + fp) * 100
print(f'Linear SVM sensitivity: {lin_sensitivity}%')
print(f'Linear SVM specificity: {lin_specificity}%')

#RBF SVM
tn, fp, fn, tp = confusion_matrix(test_labels, rbf_pred).ravel()
rbf_sensitivity = tp / (tp + fn) * 100
rbf_specificity = tn / (tn + fp) * 100
print(f'RBF SVM sensitivity: {rbf_sensitivity}%')
print(f'RBF SVM specificity: {rbf_specificity}%')

#Classification report
print(f'Linear SVM classification report:\n {classification_report(test_labels, linear_pred)}')
print(f'RBF SVM classification report:\n {classification_report(test_labels, rbf_pred)}')

#Plotting confusion matrix
#Linear SVM
ConfusionMatrixDisplay.from_predictions(
    y_true = test_labels,
    y_pred = linear_pred,
    display_labels = ['Normal', 'Pneumonia'],
    cmap = plt.cm.Blues,
    xticks_rotation='vertical'
)
plt.title('RSVM Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(result_path, 'LSVM_confusion_matrix.png'))

#RBF SVM
ConfusionMatrixDisplay.from_predictions(
    y_true = test_labels,
    y_pred = rbf_pred,
    display_labels = ['Normal', 'Pneumonia'],
    cmap = plt.cm.Blues,
    xticks_rotation='vertical'
)
plt.title('RSVM Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(result_path, 'RSVM_confusion_matrix.png'))

#Learning Curves
#Linear SVM
train_sizes, train_scores, test_scores = learning_curve(linear_model, train_images, train_labels, cv=5)
display = LearningCurveDisplay(
    train_sizes=train_sizes,
    train_scores=train_scores,
    test_scores=test_scores,
    score_name = 'Score'
)
display.plot()
plt.title("Linear SVM Learning Curve")
plt.savefig(os.path.join(result_path, 'Linear_SVM_Learning_Curve.png'))

#RBF SVM
train_sizes, train_scores, test_scores = learning_curve(rbf_model, train_images, train_labels, cv=5)
display = LearningCurveDisplay(
    train_sizes=train_sizes,
    train_scores=train_scores,
    test_scores=test_scores,
    score_name = 'Score'
)
display.plot()
plt.title("RBF SVM Learning Curve")
plt.savefig(os.path.join(result_path, 'RBF_SVM_Learning_Curve.png'))

#saving results to json file
lin_results = {
    'accuracy': accuracy_score(test_labels, linear_pred),
    'sensitivity': lin_sensitivity,
    'specificity': lin_specificity
}
rbf_results = {
    'accuracy': accuracy_score(test_labels, rbf_pred),
    'sensitivity': rbf_sensitivity,
    'specificity': rbf_specificity
}
with open(os.path.join(result_path, 'lin_result.json'), 'w') as fp:
    json.dump(lin_results, fp)
with open(os.path.join(result_path, 'rbf_result.json'), 'w') as fp:
    json.dump(rbf_results, fp)
