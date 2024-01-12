#using argparse and flags to control operations.
import sys
import os
import argparse
import configparser

#=============================Arg Parser=======================================#
'''
This section of the code is used to parse arguments from the command line. Each file
will either read directly from the config file, or these can be overwritten using
command line arguments.
'''
#initialise parser
parser = argparse.ArgumentParser(description='Run a python file with or without arguments')

#add arguments to parser
parser.add_argument('-f', "--filename", help='Name of python file to run')
parser.add_argument('-e', '--epochs', help='Number of epochs for training')
parser.add_argument('-lr', '--learning_rate', help='Learning rate for training')
parser.add_argument('-bs', '--batch_size', help='Batch size for training')
parser.add_argument('-k', '--k_neighbors', help='Number of neighbors for KNN')
parser.add_argument('-c', '--c_value', help='C value for SVM')
parser.add_argument('-g', '--gamma_value', help='Gamma value for SVM')
parser.add_argument('-n', '--n_estimators', help='Number of estimators for AdaBoost')
parser.add_argument('-a', '--algorithm', help='Algorithm for AdaBoost')
parser.add_argument('-p', '--pretrained', help = 'Use pretrained model for task')
args = vars(parser.parse_args())

if __name__ == '__main__':
    # ======================Config File Initialisation===============================#
    '''
    This section of the code is used to read from config file for default parameters.
    '''
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'filename': 'B/CNN3.py',
                         'epochs': 20,
                         'learning_rate': 0.001,
                         'batch_size': 64,
                         'k_neighbors': 5,
                         'c_value': 1,
                         'gamma_value': 0.0001,
                         'n_estimators': 100,
                         'algorithm': 'SAMME.R',
                         'pretrained': 0
                         }
    #check if user entered epoch or necessary parameters
    config['USER'] = {k: v for k, v in args.items() if v is not None}
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    os.system('python3 ' + args['filename'])
