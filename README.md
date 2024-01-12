# MLS Project Repository 2023-2024
This repository is used to document the progress made for the MLS Project 2023-2024. 

## Project Overview
As machine learning techniques grow more and more advanced, day by day, new methods and domains in which such techniques can be applied are constantly being discovered. One potential application of machine learning is in the field of medical diagnostics, where machine learning techniques may be used to classify different diseases and help doctors with arriving at diagnoses. This project aims to assess the efficacy of using machine learning models for disease diagnosis, particularly for the diagnosis of pneumonia and colorectal cancer.

## Project Brief
The project consists of two primary tasks:
1. Task A: Binary Classification of the PneumoniaMNIST dataset - determine whether a patient (input image scan) has pneumonia or not.
2. Task B: Multi-class Classification of the PathMNIST dataset - classify an image onto 9 different types of tissues based on samples of patient colons

## Repository Structure
The repository is structured as follows:

```
AMLS_23_24_SN20004671
|__A (Task A Code Folder) #contains code and results for Task A
|  |__results #contains results for task A
|  |__AdaBoostClassifier.py #contains AdaBoostClassifier model
|  |__CNN.py #contains CNN model
|  |__KNN.py #contains KNN model
|  |__LogisticRegression.py #contains Logistic Regression model
|  |__ResNet50.py #contains ResNet50 model
|  |__SVM.py #contains SVM model
|__B
|  |__results
|  |__CNN.py
|  |__ResNet50.py
|__datasets
|__pipeline
|  |__data_analysis.py
|  |__preprocessing.py
|  |__modelcomparison.py
|  config.ini #contains default and user-defined hyperparameters for models
|  main.py #main file to run tasks and models from CLI
|  README.md
|  requirements.txt #requirements to be installed
```
## Installation
To install the required packages, run the following command:
```
pip install -r requirements.txt
```
## Using the Project
This project uses flags and argparse to interpret and compile files from the command line.
The following flags are available:
```
-f (--filename) #specify filename of model to run (e.g. A/CNN.py) (required)
-e (--epochs) #specify number of epochs to run for CNN and ResNet50 (default = 20)
-b (--batch_size) #specify batch size for CNN and ResNet50 (default = 64)
-l (--learning_rate) #specify learning rate for CNN, ResNet, Decision Tree and AdaBoost 
(default = 0.001)
-k (--k_neighbors) #specify number of neighbors for KNN (default = 5)
-n (--n_estimators) #specify number of estimators for AdaBoost (default = 100)
-a (--algorithm) #specify algorithm for AdaBoost (default = SAMME.R)
-c (--c_value) #specify C value for SVM (default = 1)
-g (--gamma_value) #specify gamma value for SVM (default = 0.001)
```
To run the necessary files, please use the main file and enter system arguments to specify hyperparameters. For example,
to run a task without specifying hyperparameters, please run the following command. This will draw default arguments from the config.ini file, shown further below.
```
python3 main.py -f A/CNN.py
```
An example of running tasks with hyperparameters is as follows:
```
python3 main.py -f A/CNN.py -e 20 -b 64 -lr 0.001
```
To view the help menu, run the following command:
```
python3 main.py --help
```
The config.ini file stores the following default parameters for the models.
These values will be taken if no hyperparameters are specified.
```
['DEFAULT']
'epochs' = 20
'batch_size' = 64
'learning_rate' = 0.001
'k_neighbors' = 5
'n_estimators' = 100
'algorithm' = 'SAMME.R'
'c_value' = 1
'gamma_value' = 0.001
```
## Acknowledgements
The datasets used in this project are from the MedMNIST repository, and are cited as follows:
```
Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni. Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data, 2023.
                            
Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.
```
Additionally, the instructions and inspiration for this project arise from the ELEC0134 module, led by Dr. Miguel Rodrigues at UCL.