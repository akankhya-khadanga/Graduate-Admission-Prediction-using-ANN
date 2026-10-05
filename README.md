# Graduate Admission Prediction

A neural network-based regression model built with **TensorFlow/Keras** to predict a student's probability of admission to graduate school.

## Overview

The project uses academic and profile-related factors to estimate the **Chance of Admit**.

The notebook covers:

* Loading and exploring the admission dataset
* Removing unnecessary columns
* Splitting the data into training and testing sets
* Scaling features using Min-Max Scaling
* Building a neural network with Keras
* Training and validating the model
* Evaluating predictions using R² score

## Dataset

The project uses the **Graduate Admissions** dataset.

The features include:

* GRE Score
* TOEFL Score
* University Rating
* SOP
* LOR
* CGPA
* Research

The target variable is:

```text
Chance of Admit
```

The `Serial No.` column is removed before training as it does not provide useful predictive information.

## Data Preprocessing

The data is divided into training and testing sets using an 80/20 split.

Feature values are then scaled using `MinMaxScaler`:

```python
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

This scales the features to a common range and helps the neural network train more effectively.

## Model Architecture

```text
Input Features
      ↓
Dense(7, ReLU)
      ↓
Dense(1, Linear)
      ↓
Chance of Admit
```

The output layer uses a **linear activation** because the project is predicting a continuous numerical value rather than a class.

## Training

* **Framework:** TensorFlow / Keras
* **Optimizer:** Adam
* **Loss:** Mean Squared Error
* **Epochs:** 100
* **Validation Split:** 20%

## Evaluation

The model's predictions are evaluated using the **R² (R-squared) score**, which measures how well the model explains the variation in the target variable.

Training and validation loss are also plotted to observe the model's performance during training.

## Technologies

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

## Project Structure

```text
Graduate-Admission-Prediction/
├── notebook98205f5e7a.ipynb
└── README.md
```

## How to Run

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

Open the notebook in **Jupyter Notebook, Google Colab, Kaggle, or VS Code** and run the cells.

## What I Learned

This project helped me understand how neural networks can be applied to **regression problems**, including feature scaling, model architecture, loss functions, validation, and R²-based evaluation.

## Future Improvements

* Experiment with different neural network architectures
* Tune the learning rate and number of neurons
* Compare the neural network with traditional regression models
* Add additional evaluation metrics such as MAE and RMSE

