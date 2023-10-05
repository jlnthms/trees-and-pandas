# Trees & Pandas

A Python decision tree and random forest implementation using only Pandas.

![pandas](https://github.com/jlnthms/numpy-neural-network/assets/74052135/5bf52fd8-fe0c-45ec-b96c-c32db56ae38b)

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Upcoming Extensions](#upcoming-extensions)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Author](#author)

## Introduction

This project provides an educational implementation of a Decision Tree classifier from scratch using only pandas as data 
structure. This project was primarily designed for personal learning and comprehension of decision trees, using
insightful object-oriented architecture for enhanced understanding of the theory beneath one of the most common 
machine learning algorithms.

## Key Features

- **Pandas input & output:** The model expects a Dataframe and will return a Dataframe, while using custom data handling.
- **Tree representation:** The tree, once built, can be either printed or plotted.

## Upcoming Extensions

- **Random Forest:** Similar Random forest package using DecisionTree is being developed.

## Getting Started

### Prerequisites

To run this project, you'll need:

- Python (3.7+ recommended)
- Pandas
- Jupyter (& additional dependencies)

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/jlnthms/trees-and-pandas.git
   cd trees-and-pandas
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

The notebook *test.ipynb* already provides a concrete runnable use case of fitting the tree on the iris dataset 
for classification, however here is a more generic example:
   
```python
# Import necessary modules
import sys
sys.path.append('../')
import pandas as pd
from Dataset.dataset import *
from DecisionTree.tree import DecisionTree

from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load your data into a dataframe and create a Dataset object
X, y = ...
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2)
df = pd.concat([X_train,y_train], axis=1).reset_index(drop=True)
label_column = 'your_label'
dataset = Dataset(df, label_column)

# Fit a tree to your training data
tree = DecisionTree(dataset)
max_depth, min_samples = ... # select hyperparameters
tree.fit(max_depth, min_samples)

# and make predictions on the test data
predictions = tree.predict(X_test)
print(predictions)

```

## Author

**Julien Thomas**
- GitHub: [JulienThomas](https://github.com/jlnthms)
- Email: thomasjulien92140@gmail.com
- LinkedIn: https://www.linkedin.com/in/julien-thomas-826b4920b/