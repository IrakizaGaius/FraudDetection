# Detecting Fraudulent Transactions with Machine Learning

## Project Overview

This project delves into detecting fraudulent online transactions using both **classical machine learning models** and  **deep neural networks** . My main goal was to explore and compare how well these different approaches perform when applying various optimization techniques. The model uses a comprehensive dataset combining transaction and identity details to predict if a transaction is fraudulent.

The dataset features a mix of  **categorical and numerical data** , where identity and transaction information are linked by `TransactionID`. The core task is a binary classification: identifying whether a transaction is legitimate or fraudulent.

---

## Dataset Details

My analysis relies on a dataset compiled from two primary files:

* **`train_transaction.csv`** and  **`train_identity.csv`** , which are joined using `TransactionID`.
* The features span various aspects, including **transaction specifics** (e.g., card, address, email details) and **identity attributes** (e.g., device type, user IDs).
* The objective is to predict the **probability of fraud** (`isFraud` = 1) for entries in the test set.

### Steps to Access the Dataset

1. **Go to the competition page** :

* Visit: [https://www.kaggle.com/competitions/ieee-fraud-detection/](https://www.kaggle.com/competitions/ieee-fraud-detection/)

2. **Sign in or Sign up** :

* If you don’t have a Kaggle account, you’ll need to create one.
* Accept the competition rules (required to download the data).

3. **Download the data** :

* On the competition page, go to the **“Data”** tab.
* Download the following files:
  * `train_transaction.csv`
  * `train_identity.csv`
  * `test_transaction.csv`
  * `test_identity.csv`
  * `sample_submission.csv`

### Note

* The dataset is large (~1.35 GB), so make sure you have enough space.
* You’ll be predicting `isFraud`, a binary classification target.
* The data is split into transaction and identity tables joined by `TransactionID`.

---

## Models

### LightGBM Model Configuration & Performance Summary

This table summarizes the **key hyperparameters** used in training a LightGBM classifier for fraud detection and the resulting **performance metrics** evaluated on the  **test set** . The model was tuned to handle class imbalance (class_weight='balanced') and trained with a relatively deep tree structure to capture complex patterns in the data.

| **Hyperparameter** | **Value** | **Metric**      | **Score** |
| ------------------------ | --------------- | --------------------- | --------------- |
| n_estimators             | 5500            | Accuracy              | 0.99            |
| learning_rate            | 0.05            | Precision (Not Fraud) | 0.99            |
| max_depth                | 8               | Recall (Not Fraud)    | 0.99            |
| num_leaves               | 64              | F1-Score (Not Fraud)  | 0.99            |
| class_weight             | balanced        | Precision (Fraud)     | 0.84            |
| random_state             | 42              | Recall (Fraud)        | 0.78            |
| n_jobs                   | -1              | F1-Score (Fraud)      | 0.81            |

### Neural Network Optimization Experiments

I conducted a series of experiments to optimize my neural network models. The table below summarizes the configurations tested, with performance metrics to be populated upon completion of training on the test split of the dataset.

**Disclaimer:**

All the metrics recorded below Focus on the ability of the model to detect **fraudulent transactions on the test data of my dataset.** since due to a very high imbalance of my dataset which also reflect real transactions made. I Found out that it would be meaningless to record the **Not Fraud metrics** due to it having the highest recurring data which makes the model usually inflated due to class imbalance and aren't helpful for my model evaluation focus.

| Instance | Optimizer | Regularizer                                                              | Epochs      | Early Stopping                    | Layers | Learning Rate | Accuracy(Test data) | F1-score(Test data) | Precision(Test data) | Recall(Test data) |
| :------- | :-------- | :----------------------------------------------------------------------- | :---------- | :-------------------------------- | :----- | :------------ | :------------------ | :------------------ | :------------------- | :---------------- |
| 1        | Default   | Default                                                                  | Default (1) | No                                | 6      | Default       | 0.97                | 0.38                | 0.87                 | 0.24              |
| 2        | Adam      | L2(0.0001)on every hidden layer                                          | 50          | Yes (Patience = 10 on val_loss)  | 6      | 0.0001        | 0.98                | 0.63                | 0.82                 | 0.51              |
| 3        | RMSprop   | L1,<br />Dropout[0.2, 0.1, 0.1, 0.05, 0.0] After each Layer respectively | 100         | Yes (Patience = 10 on val_loss)  | 6      | 0.0001        | 0.97                | 0.47                | 0.80                 | 0.33              |
| 4        | SGD       | L2                                                                       | 150         | Yes (Patience = 10 on val_loss) | 6      | 0.01          | 0.98                | 0.55                | 0.81                 | 0.42              |
| 5        | Nadam     | L2                                                                       | 300         | Yes (Patience = 10 on val_loss) | 6      | 0.005         | 0.98                | 0.64                | 0.80                 | 0.53              |

## Findings Summary

### Best Neural Network Performance

The most effective neural network configuration was  **Instance 5** , which achieved the **highest F1-score** on the test set — the most reliable indicator in imbalanced classification tasks. This model managed to balance both **precision** and  **recall** , making it more practical for real-world fraud detection use, where catching frauds (recall) without triggering too many false alarms (precision) is key.

**Top Model (Instance 5):**

* **Optimizer:** Nadam
* **Regularization:** L2
* **Epochs:** Up to 300 (Early Stopping applied)
* **Early Stopping:** Enabled (Patience = 10 on `val_loss`)
* **Layers:** 6
* **Learning Rate:** 0.005
* **Test Metrics:**
  * **F1-score:** 0.64 (Best)
  * **Precision:** 0.80
  * **Recall:** 0.53
  * **Accuracy:** 0.98

### Conclusion

The experiments demonstrated that  **model architecture** ,  **regularization** , and particularly the **choice of optimizer and learning rate** significantly impacted performance. Among all instances, **Instance 5** achieved the best fraud detection trade-off. It outperformed other models in terms of  **F1-score** , meaning it struck a better balance between catching fraudulent cases and avoiding false alarms.

Although accuracy remained high across all models due to class imbalance, it was **not a reliable indicator** for model effectiveness in fraud detection. Instead, **Instance 5's balance of high precision and reasonable recall** indicates it is well-suited for practical deployment or further tuning for production scenarios.

### Comparison between the lightGBM model and Neural network best performer model

In evaluating multiple models on the highly imbalanced IEEE fraud detection dataset, LightGBM (Instance 6) outperformed all neural network architectures tested, including the best-performing neural net (Instance 5), by achieving an outstanding F1-score of 0.8120, with precision of 0.84 and recall of 0.78. While Instance 5, a deep neural network with Nadam optimizer, L2 regularization, and batch normalization, showed solid results (F1-score of 0.6358), it was ultimately surpassed by LightGBM’s superior handling of class imbalance and structured data. LightGBM’s gradient boosting framework naturally focused learning on minority (fraudulent) cases and captured complex patterns with minimal tuning, making it the most effective model for fraud detection in this study.

### Data Splitting Strategy

The dataset was divided into the following proportions for training, validation, and testing:

* **Training Set:** 70%
* **Validation Set:** 15%
* **Test Set:** 15%

---

## Saved Models

All trained models are stored in the `saved_models/` directory:

* `nn_instance1.keras`
* `nn_instance2.keras`
* `nn_instance3.keras`
* `nn_instance4.keras`
* `lightgbm_model.joblib`

---

## How To run this Project

This project builds and evaluates various machine learning models, including LightGBM and neural networks, to detect fraudulent transactions.

### Getting Started

Follow these steps to set up your environment and run the code.

### 1. Clone the Repository

**1. git clone [https://github.com/IrakizaGaius/FraudDetection.git](https://github.com/IrakizaGaius/FraudDetection.git)**

cd Fraud Detectionetection

**2. Create and Activate a Virtual Environment (Recommended)**
Create a virtual environment to manage dependencies:

python -m venv .venv

**3. Activate the virtual environment:**

**On macOS/Linux** :

source .venv/bin/activate

**On Windows** :

.venv\Scripts\activate
**4. Install Dependencies**

Install the required libraries using:

pip install -r requirements.txt
**5. Run the Project**

After setting up your environment, you can execute the training scripts or notebooks provided to train and evaluate the models.

**Models Used**:
**LightGBM** with tuned hyperparameters
**Neural Networks** (with different configurations including dropout, regularization, and optimizers)

**Evaluation Metrics**
Accuracy
Precision
Recall
F1-Score
Confusion Matrix

---

## Project Walkthrough Video

A 5-minute video presentation covering the dataset summary, model architectures, optimization strategies, and performance comparisons will be linked here:

📹 [Link to Video Presentation]
