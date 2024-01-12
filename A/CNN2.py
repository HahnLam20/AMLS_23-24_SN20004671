#General Imports
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
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate, GridSearchCV, learning_curve, LearningCurveDisplay, validation_curve, ValidationCurveDisplay

#torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.transforms import transforms, v2
from torchinfo import summary

#medmnist imports
import medmnist
from medmnist import INFO, evaluator

#Plotting Style
plt.style.use(['science', 'no-latex'])  #Needs LaTeX installed. If not, comment out and use plt.style.use(['science', 'no-latex']) instead.

#check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

#initialising dataflag
data_flag = 'pneumoniamnist'

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
result_path = os.path.join(path, 'A/results/CNN2')
model_path = os.path.join(path, 'models/A')

#establishing connection to pipeline
pipeline_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(pipeline_dir, '..'))
from pipeline.preprocessing import preprocess
from pipeline.resultsanalysis import gen_loss_plot, gen_acc_plot, gen_confusion_matrix, gen_json
from pipeline.utils import clean_folder
clean_folder(result_path)

#initialise dataset for visualisation
pneumdata = np.load(os.path.join(dataset_path, 'pneumoniamnist.npz'))

#extracting images and labels from dataset
train_images = pneumdata['train_images']
val_images = pneumdata['val_images']
test_images = pneumdata['test_images']
train_labels = pneumdata['train_labels']
val_labels = pneumdata['val_labels']
test_labels = pneumdata['test_labels']

#preprocess and encapsulation
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = DataClass(root=dataset_path, split='train', transform=data_transform, download=False)
test_dataset = DataClass(root=dataset_path, split='test', transform=data_transform, download=False)
val_dataset = DataClass(root=dataset_path, split='val', transform=data_transform, download=False)

#initialise dataloaders
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

print(train_dataset)
print("=========================================")
print(test_dataset)

#check if load saved model or not


#implement CNN2 model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride = 1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #flatten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



if pretrained:
    model = CNN().to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, 'cnn2.pth')))
    print(summary(model, input_size=(batch_size, n_channels, 28, 28)))
else:
    model = CNN().to(device)
    print(summary(model, input_size=(batch_size, n_channels, 28, 28)))
    #defining loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #training model
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

    #save trained model
    torch.save(model.state_dict(), os.path.join(model_path, 'cnn2.pth'))

#plotting training and validation loss
gen_loss_plot(train_loss, val_loss, result_path)


#plotting training and validation accuracy+
gen_acc_plot(train_acc, val_acc, result_path)



#Make predictions and saving as csv
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
test_labels = test_dataset.labels.squeeze()
incorrect_preds = np.where(preds != test_labels)[0]
correct_preds = np.where(preds == test_labels)[0]
accuracy = int(correct_preds.shape[0]) / preds.shape[0] * 100
print(f"Prediction Accuracy: {accuracy}%")
plt.figure(figsize=(10,7))
for i in range(n_classes):
    plt.bar(i, np.sum(test_labels[correct_preds] == i), color = 'g')
    plt.bar(i, np.sum(test_labels[incorrect_preds] == i), bottom=np.sum(test_labels[correct_preds] == i), color = 'r')
plt.xticks(np.arange(n_classes), info['label'])
plt.xlabel('Class Label')
plt.ylabel('Count')
plt.title('Predictions as a Function of Class Label')
plt.legend(['Correct', 'Incorrect'])
plt.savefig(os.path.join(result_path, 'predictions_as_function_of_class_label.png'))

#confusion matrix
gen_confusion_matrix(test_labels, preds, result_path, data_flag)

#calculation of sensitivity and specificity
tn, fp, fn, tp = confusion_matrix(test_labels, preds).ravel()
sensitivity = tp / (tp + fn) * 100
specificity = tn / (tn + fp) * 100
print(f'Sensitivity: {sensitivity}%')
print(f'Specificity: {specificity}%')

#Saving results to JSON
gen_json(accuracy, sensitivity, specificity, result_path)
