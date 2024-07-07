# 1-Iris-Dataset
Comprehensive analysis and visualization of the Iris dataset, including data preprocessing, exploratory data analysis, and machine learning modeling using a decision tree classifier. This README will include sections commonly found in data analysis and visualization projects, such as an introduction, installation instructions, usage, and more.

---

# Iris Dataset Analysis and Visualization

This repository contains the analysis and visualization of the famous Iris dataset. The Iris dataset is widely used for machine learning and data visualization practice.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis](#analysis)
- [Visualization](#visualization)
- [Modeling](#modeling)
- [Results](#results)


## Introduction

The Iris dataset consists of 150 samples from each of three species of Iris flowers (Iris setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the lengths and the widths of the sepals and petals.

## Dataset

The dataset used in this project can be found on [Kaggle](https://www.kaggle.com/uciml/iris). It contains the following columns:

- Id
- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm
- Species

## Installation

To get started with the project, clone this repository and install the required dependencies:

```bash
git clone https://github.com/your-username/iris-dataset-analysis.git
cd iris-dataset-analysis
pip install -r requirements.txt
```

## Usage

The main analysis and visualization are provided in the Jupyter Notebook file. To run the notebook:

```bash
jupyter notebook Iris_Dataset_Analysis.ipynb
```

## Analysis

In this section, we perform basic data analysis on the Iris dataset. We explore the data, handle missing values, and perform statistical analysis to understand the distribution and relationships between different features.

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('Iris.csv')

# Display the first few rows of the dataset
df.head()
```

## Visualization

We use various visualization techniques to understand the data better:

- Histograms
- Scatter plots
- Pair plots
- Box plots

These visualizations help in understanding the distribution and relationship between the features.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Pair plot
sns.pairplot(df, hue='Species')
plt.show()
```

## Modeling

We build a decision tree classifier to classify the species of Iris flowers based on their features. We train the model and evaluate its performance.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the dataset
X = df.drop(columns=['Species'])
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
```

## Results

The decision tree classifier achieves an accuracy of 100% on the test dataset. Below is the confusion matrix showing the model's performance:

```
array([[19,  0,  0],
       [ 0, 13,  0],
       [ 0,  0, 13]], dtype=int64)
```

