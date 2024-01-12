#Displaying the different class distributions within the datasets.

# General Imports
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# Plotting Style # Needs LaTeX installed. If not, comment out and use plt.style.use(['science', 'no-latex']) instead.

#initialising datasets
pneumdata = np.load('../datasets/pneumoniamnist.npz')
pathdata = np.load('../datasets/pathmnist.npz')

#Analysis of datasets by labels
print("Pneumonia Dataset")
print("Training labels: {}".format(pneumdata['train_labels']))
normal_count = pneumdata['train_labels'].tolist().count([0]) + pneumdata['val_labels'].tolist().count([0]) + pneumdata['test_labels'].tolist().count([0])
pneumonia_count = pneumdata['train_labels'].tolist().count([1]) + pneumdata['val_labels'].tolist().count([1]) + pneumdata['test_labels'].tolist().count([1])
print("Number of normal images: {}".format(normal_count))
print("Number of pneumonia images: {}".format(pneumonia_count))
print("Majority class: Pneumonia. Percentage: {}%".format(pneumonia_count/(normal_count+pneumonia_count)*100))

print("Colorectal Cancer Dataset")
print("Training labels: {}".format(pathdata['train_labels']))
adipose_count = pathdata['train_labels'].tolist().count([0]) + pathdata['val_labels'].tolist().count([0]) + pathdata['test_labels'].tolist().count([0])
background_count = pathdata['train_labels'].tolist().count([1]) + pathdata['val_labels'].tolist().count([1]) + pathdata['test_labels'].tolist().count([1])
debris_count = pathdata['train_labels'].tolist().count([2]) + pathdata['val_labels'].tolist().count([2]) + pathdata['test_labels'].tolist().count([2])
lymphocytes_count = pathdata['train_labels'].tolist().count([3]) + pathdata['val_labels'].tolist().count([3]) + pathdata['test_labels'].tolist().count([3])
mucus_count = pathdata['train_labels'].tolist().count([4]) + pathdata['val_labels'].tolist().count([4]) + pathdata['test_labels'].tolist().count([4])
smooth_muscle_count = pathdata['train_labels'].tolist().count([5]) + pathdata['val_labels'].tolist().count([5]) + pathdata['test_labels'].tolist().count([5])
normal_colon_mucosa_count = pathdata['train_labels'].tolist().count([6]) + pathdata['val_labels'].tolist().count([6]) + pathdata['test_labels'].tolist().count([6])
cancer_associated_count = pathdata['train_labels'].tolist().count([7]) + pathdata['val_labels'].tolist().count([7]) + pathdata['test_labels'].tolist().count([7])
colorectal_adenocarcinoma_count = pathdata['train_labels'].tolist().count([8]) + pathdata['val_labels'].tolist().count([8]) + pathdata['test_labels'].tolist().count([8])
print("Number of adipose images: {}".format(adipose_count))
print("Number of background images: {}".format(background_count))
print("Number of debris images: {}".format(debris_count))
print("Number of lymphocytes images: {}".format(lymphocytes_count))
print("Number of mucus images: {}".format(mucus_count))
print("Number of smooth muscle images: {}".format(smooth_muscle_count))
print("Number of normal colon mucosa images: {}".format(normal_colon_mucosa_count))
print("Number of cancer-associated stroma images: {}".format(cancer_associated_count))
print("Number of colorectal adenocarcinoma epithelium images: {}".format(colorectal_adenocarcinoma_count))

print("Majority Class: {}. Percentage: {}%".format(max(adipose_count, background_count, debris_count, lymphocytes_count, mucus_count, smooth_muscle_count, normal_colon_mucosa_count, cancer_associated_count, colorectal_adenocarcinoma_count), max(adipose_count, background_count, debris_count, lymphocytes_count, mucus_count, smooth_muscle_count, normal_colon_mucosa_count, cancer_associated_count, colorectal_adenocarcinoma_count)/(adipose_count+background_count+debris_count+lymphocytes_count+mucus_count+smooth_muscle_count+normal_colon_mucosa_count+cancer_associated_count+colorectal_adenocarcinoma_count)*100))
print("Minority Class: {}. Percentage: {}%".format(min(adipose_count, background_count, debris_count, lymphocytes_count, mucus_count, smooth_muscle_count, normal_colon_mucosa_count, cancer_associated_count, colorectal_adenocarcinoma_count), min(adipose_count, background_count, debris_count, lymphocytes_count, mucus_count, smooth_muscle_count, normal_colon_mucosa_count, cancer_associated_count, colorectal_adenocarcinoma_count)/(adipose_count+background_count+debris_count+lymphocytes_count+mucus_count+smooth_muscle_count+normal_colon_mucosa_count+cancer_associated_count+colorectal_adenocarcinoma_count)*100))

#plotting distribution of class images as pi charts
#Pneumonia Dataset
fig, ax = plt.subplots()
ax.pie([normal_count, pneumonia_count], labels = ['Normal', 'Pneumonia'], autopct='%1.1f%%', startangle=90)
ax.axis('equal')
ax.set_title('Distribution of Pneumonia Dataset')
plt.savefig('../A/results/pneumonia_distribution.png')

#Colorectal Cancer Dataset
fig, ax = plt.subplots()
ax.pie([adipose_count, background_count, debris_count, lymphocytes_count, mucus_count, smooth_muscle_count, normal_colon_mucosa_count, cancer_associated_count, colorectal_adenocarcinoma_count], labels = ['Adipose', 'Background', 'Debris', 'Lymphocytes', 'Mucus', 'Smooth Muscle', 'Normal Colon Mucosa', 'Cancer-Associated Stroma', 'Colorectal Adenocarcinoma'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
ax.set_title('Distribution of Colorectal Cancer Dataset')
plt.savefig('../B/results/colorectal_distribution.png')
