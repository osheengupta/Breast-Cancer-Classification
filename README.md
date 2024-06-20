# Machine Learning : Breast Cancer Classification

## Table of Contents
* [Introduction](#introduction)
* [Dataset](#dataset)
* [Installation](#installation)
* [Screenshots](#screenshots)
* [Process](#process)
* [Code Examples](#code-examples)
* [Status](#status)
* [Contact](#contact)

## Introduction
This project focuses on the classification of breast cancer using machine learning techniques. The goal is to develop a model that can accurately classify whether a breast cancer tumor is benign or malignant based on various features.


## Dataset

The dataset used in this project is the <a href="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic" target="_blank">Breast Cancer Wisconsin (Diagnostic) </a> Data Set. It includes features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

* Number of instances: 569
* Number of attributes: 30
* Class Distribution: 212 Malignant, 357 Benign
* Target class: - Malignant - Benign
* 30 features are used, examples: - radius (mean of distances from center to points on the perimeter) - texture (standard deviation of gray-scale values) - perimeter - area - smoothness (local variation in radius lengths) - compactness (perimeter^2 / area - 1.0) - concavity (severity of concave portions of the contour) - concave points (number of concave portions of the contour) - symmetry - fractal dimension ("coastline approximation" - 1)

## Installation

To run this project, you need to have Python installed along with the required libraries. You can install the required libraries using:

```
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

```

## Screenshots
### Correlation among variables
![co](./img2/Picture1.png)

### Heatmap of correlation
![Heatmap](./img2/Picture2.png)

### Heatmap of Confusion matrix (Using scaled data and best parameters)
![CM](./img2/Picture3.png)

### Accuracy measures
![AM](./img2/Picture4.png)

## Process

1. Load the Dataset: The dataset is loaded from the UCI Machine Learning Repository.It can also be found <a href="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic" target="_blank">here </a>.
2. Visualizing the Data: Correlation between various features are displayed with the help of scatter plots.
3. Training the model: I have used Support Vector Model (SVM) for making the predictions.
4. Split the Dataset: The dataset is split into training and testing sets.
5. Evaluate Models: The model is trained and evaluated on the test set. The evaluation metrics include accuracy, precision, recall, and F1-score.
6. Improving the model:
   * Standardize the Features: Standardization ensures that the features have a mean of 0 and a standard deviation of 1.
   * Using GridsearchCV: Best parameteres were found (C & gamma)
7. Final Model: Build a model on the scaled data using the best parameters.  


## Code Examples

A few examples of useful commands/code snippets.

### Support Vector Model
```
X = df_cancer.drop(['target'],axis=1)
y = df_cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)

svc_model = SVC()
svc_model.fit(X_train, y_train)

y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
```

## Status

Project is: _finished_.

## Contact
If you loved what you read here and feel like we can collaborate to produce some exciting stuff, or if you
just want to shoot a question, please feel free to connect with me on <a href="osheengupta1994@gmail.com" target="_blank">email</a> or
<a href="www.linkedin.com/in/osheengupta" target="_blank">LinkedIn</a>
