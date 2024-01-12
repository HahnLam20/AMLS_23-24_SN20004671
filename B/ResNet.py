#General Imports
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from PIL import Image as im
import urllib
import warnings
import scienceplots
import configparser
import json
warnings.filterwarnings("ignore")

#sklearn imports
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import cross_validate, GridSearchCV, learning_curve, LearningCurveDisplay

#torch imports
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torchvision.transforms import transforms, v2

#medmnist imports
import medmnist
from medmnist import INFO, evaluator

#Modifying plot style
plt.style.use(['science', 'no-latex'])  #Needs LaTeX installed. If not, comment out and use plt.style.use(['science', 'no-latex']) instead.

#check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

#initialising dataflag
data_flag = 'pathmnist'

#initialise parameters from config file
config = configparser.ConfigParser()
config.read('config.ini')
try:
    epochs = int(config['USER']['epochs'])
except(KeyError):
    epochs = int(config['DEFAULT']['epochs'])
try:
    lr = float(config['USER']['learning_rate'])
except(KeyError):
    lr = float(config['DEFAULT']['learning_rate'])
try:
    batch_size = int(config['USER']['batch_size'])
except(KeyError):
    batch_size = int(config['DEFAULT']['batch_size'])
try:
    pretrained = bool(int(config['USER']['pretrained']))
except(KeyError):
    pretrained = bool(int(config['DEFAULT']['pretrained']))

print("Pretrained model = {}".format(pretrained))
print("Using: batch_size = {}, lr = {}, epochs = {}".format(batch_size, lr, epochs))

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])

#getting path for main directory for dataset and results
path = os.getcwd()
dataset_path = os.path.join(path, 'datasets')
result_path = os.path.join(path, 'B/results/ResNet50')
model_path = os.path.join(path, 'models/B')

#asserting connection to pipeline
pipeline_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(pipeline_dir, '..'))
from pipeline.preprocessing import preprocess
from pipeline.resultsanalysis import gen_loss_plot, gen_acc_plot, gen_confusion_matrix, gen_json
from pipeline.utils import clean_folder
clean_folder(result_path)

#initialise dataset for visualisation
pathdata = np.load(os.path.join(dataset_path, 'pathmnist.npz'))

#extracting images and labels from dataset
train_images = pathdata['train_images']
val_images = pathdata['val_images']
test_images = pathdata['test_images']
train_labels = pathdata['train_labels']
val_labels = pathdata['val_labels']
test_labels = pathdata['test_labels']

#preprocessing and encapsulation
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = DataClass(root=dataset_path, split='train', transform=data_transform, download=False)
test_dataset = DataClass(root=dataset_path, split='test', transform=data_transform, download=False)
val_dataset = DataClass(root=dataset_path, split='val', transform=data_transform, download=False)

train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=True)

#Loading model
model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

#freezing layers
for param in model.parameters():
    param.requires_grad = False

#changing last layer
model.fc = nn.Linear(2048, 9)

#moving model to device
model.to(device)


#Training and validating model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loss = []
train_acc = []
val_loss = []
val_acc = []
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch+1, epochs))
    print('-' * 10)

    #training
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.squeeze()
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc.cpu().numpy())
    print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    #validation
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.squeeze()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)
    val_loss.append(epoch_loss)
    val_acc.append(epoch_acc.cpu().numpy())
    print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
print('Finished Training')

#saving model
torch.save(model.state_dict(), os.path.join(model_path, 'ResNet50B.pth'))


#plotting loss
gen_loss_plot(train_loss, val_loss, result_path)

#plotting accuracy
gen_acc_plot(train_acc, val_acc, result_path)

#model evaluation
model.eval()
preds = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)
        preds.append(pred.cpu().numpy())
preds = np.concatenate(preds)
print(preds.shape)


#plot of incorrect predictions as function of classes
# test_labels = test_dataset.labels.squeeze()
test_labels = test_dataset.labels.squeeze()
print("preds.shape", preds.shape)
print("test_labels.shape", test_labels.shape)
incorrect_preds = np.where(preds != test_labels)[0]
correct_preds = np.where(preds == test_labels)[0]
print('correct_preds.shape = {}'.format(correct_preds.shape[0]))
print('preds.shape = {}'.format(preds.shape[0]))
accuracy = int(correct_preds.shape[0]) / preds.shape[0] * 100
print(f"Prediction Accuracy: {accuracy}%")
plt.figzure(figsize=(10,7))
for i in range(n_classes):
    print(f"Class: {i}, Correct: {np.sum(test_labels[correct_preds] == i)}, Incorrect: {np.sum(test_labels[incorrect_preds] == i)}")
    plt.bar(i, np.sum(test_labels[correct_preds] == i), color = 'g')
    plt.bar(i, np.sum(test_labels[incorrect_preds] == i), bottom=np.sum(test_labels[correct_preds] == i), color = 'r')
plt.xticks(np.arange(n_classes), info['label'])
plt.xlabel('Class Label')
plt.ylabel('Count')
plt.title('Predictions as a Function of Class Label')
plt.legend(['Correct', 'Incorrect'])
plt.savefig(os.path.join(result_path, 'predictions_as_function_of_class_label.png'))


#plotting confusion matrix with confusion matrix display
gen_confusion_matrix(test_labels, preds, result_path, data_flag)


# #Extracting true positive, true negative, false positive and false negative values for sensitivity and specificity
cm = confusion_matrix(test_labels, preds)
# #false positives = sum of false classification of 8
fp = np.sum(cm[8:9, 0:8])
# #false negatives = sum of false classification of 0 to 7
fn = np.sum(cm[0:8, 8:9])
# #true positives = sum of true classification of 8
tp = np.sum(cm[8:9, 8:9])
# #true negatives = sum of true classification of 0 to 7
tn = np.sum(cm[0:8, 0:8])
# #calculation of sensitivity and specificity
sensitivity = tp / (tp + fn) * 100
specificity = tn / (tn + fp) * 100
print('Sensitivity: {}%'.format(sensitivity))
print('Specificity: {}%'.format(specificity))

#Saving results to JSON
gen_json(accuracy, sensitivity, specificity, result_path)
