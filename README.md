# Detecting Fraudulent Transactions with Machine Learning

## Project Overview

This project delves into detecting fraudulent online transactions using both **classical machine learning models** and  **deep neural networks** . Our main goal was to explore and compare how well these different approaches perform when applying various optimization techniques. The model uses a comprehensive dataset combining transaction and identity details to predict if a transaction is fraudulent.

The dataset features a mix of  **categorical and numerical data** , where identity and transaction information are linked by `TransactionID`. The core task is a binary classification: identifying whether a transaction is legitimate or fraudulent.

---

## Dataset Details

Our analysis relies on a dataset compiled from two primary files:

* **`train_transaction.csv`** and  **`train_identity.csv`** , which are joined using `TransactionID`.
* The features span various aspects, including **transaction specifics** (e.g., card, address, email details) and **identity attributes** (e.g., device type, user IDs).
* The objective is to predict the **probability of fraud** (`isFraud` = 1) for entries in the test set.

---

## Neural Network Optimization Experiments

We conducted a series of experiments to optimize our neural network models. The table below summarizes the configurations tested, with performance metrics to be populated upon completion of training.

| Instance | Optimizer | Regularizer | Epochs  | Early Stopping | Layers | Learning Rate | Accuracy | F1-score | Precision | Recall |
| :------- | :-------- | :---------- | :------ | :------------- | :----- | :------------ | :------- | :------- | :-------- | :----- |
| 1        | -         | -           | Default | No             | 2      | Default       |          |          |           |        |
| 2        | Adam      | L2          | 20      | Yes            | 3      | 0.001         |          |          |           |        |
| 3        | RMSprop   | L1          | 30      | No             | 4      | 0.0005        |          |          |           |        |
| 4        | Adam      | L1 + L2     | 15      | Yes            | 5      | 0.0001        |          |          |           |        |
| 5        | SGD       | L2          | 25      | No             | 3      | 0.01          |          |          |           |        |

Export to Sheets

---

## Findings Summary

### Neural Network Performance

Our most effective neural network configuration, designated as  **Instance 2** , leveraged the following parameters:

* **Optimizer:** Adam
* **Regularization:** L2
* **Epochs:** 20
* **Early Stopping:** Enabled
* **Layers:** 3
* **Learning Rate:** 0.001

### Model Type Comparison

Here's how the best-performing models from different categories stack up:

| Model Type               | Accuracy | F1-score | Comments                           |
| :----------------------- | :------- | :------- | :--------------------------------- |
| **Neural Network** | X.XX     | X.XX     | Performed better with fine-tuning  |
| Logistic Regression      | X.XX     | X.XX     | Solid baseline, but less precise   |
| **XGBoost**        | X.XX     | X.XX     | Emerged as the top classical model |

Export to Sheets

---

## Data Splitting Strategy

We divided the dataset into the following proportions for training, validation, and testing:

* **Training Set:** 70%
* **Validation Set:** 15%
* **Test Set:** 15%

---

## Saved Models

All trained models are stored in the `saved_models/` directory:

* `nn_instance1.h5`
* `nn_instance2.h5`
* `nn_instance3.h5`
* `nn_instance4.h5`
* `xgboost_model.pkl`

---

## How to Run This Project

To set up and run the project:

1. **Clone the repository:**
   **Bash**

   ```
   git clone <your-repo-link>
   cd <project-directory>
   ```
2. **Install dependencies:**
   **Bash**

   ```
   pip install -r requirements.txt
   ```
3. **Launch the notebook:**
   **Bash**

   ```
   jupyter notebook notebook.ipynb
   ```
4. **To load a saved model** (e.g., `nn_instance2.h5`):
   **Python**

   ```
   from tensorflow.keras.models import load_model
   model = load_model('saved_models/nn_instance2.h5')
   ```

---

## Project Walkthrough Video

A 5-minute video presentation covering the dataset summary, model architectures, optimization strategies, and performance comparisons will be linked here:

📹 [Link to Video Presentation]
