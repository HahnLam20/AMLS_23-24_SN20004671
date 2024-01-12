import matplotlib.pyplot as plt
import os
import json

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from medmnist import INFO

def gen_loss_plot(train_loss, val_loss, result_path):
    '''
        Function to generate plot of losses vs epochs

        Args:
            train_loss: array of losses on the training dataset
            val_loss: array of losses on the validation dataset
            result_path: path to save the generated plot
    '''
    plt.figure(figsize=(10, 4))
    plt.plot(train_loss, label='Train loss')
    plt.plot(val_loss, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss against Epochs')
    plt.legend()
    plt.savefig(os.path.join(result_path, 'CNN2_training_and_val_loss.png'))

def gen_acc_plot(train_acc, val_acc, result_path):
    '''
        Function to generate plot of accuracies vs epochs

        Args:
            train_acc: array of accuracies on the training dataset
            val_acc: array of accuracies on the validation dataset
            result_path: path to save the generated plot
    '''
    plt.figure(figsize=(10, 4))
    plt.plot(train_acc, label='Train accuracy')
    plt.plot(val_acc, label='Val accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy against Epochs')
    plt.legend()
    plt.savefig(os.path.join(result_path, 'training_and_val_acc.png'))

def gen_confusion_matrix(test_labels, preds, result_path, data_flag):
    '''
        Function to generate confusion matrix

        Args:
            y_true: array of true labels
            y_pred: array of predicted labels
            result_path: path to save the generated plot
    '''
    info = INFO[data_flag]
    labels = list(info['label'].keys())
    ConfusionMatrixDisplay.from_predictions(
        y_true = test_labels,
        y_pred = preds,
        display_labels = labels,
        cmap = plt.cm.Blues,
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(result_path, 'CNN3_confusion_matrix.png'))

def gen_json(accuracy, sensitivity, specificity, result_path):
    '''
        Function to generate json file containing accuracy, sensitivity and specificity

        Args:
            accuracy: accuracy of the model
            sensitivity: sensitivity of the model
            specificity: specificity of the model
            result_path: path to save the generated json file
    '''
    results = {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }
    with open(os.path.join(result_path, 'result.json'), 'w') as f:
        json.dump(results, f, indent=4)
