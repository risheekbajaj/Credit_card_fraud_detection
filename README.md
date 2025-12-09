# 🛡️ FraudGuard AI: Credit Card Fraud Detection System

**A production-ready Machine Learning pipeline and interactive dashboard for detecting fraudulent credit card transactions in real-time.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red) ![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Project Overview

**FraudGuard AI** is an end-to-end machine learning project designed to identify fraudulent credit card transactions. It tackles the challenge of **highly imbalanced data** (where fraud cases are rare) using a robust Logistic Regression model.

The project goes beyond just a script—it includes a full **interactive frontend** built with Streamlit, allowing non-technical users to simulate transactions and visualize the model's decision-making process in real-time.

### 🌟 Key Features
* **Production-Grade Pipeline:** Modular code for data ingestion, preprocessing, training, and model serialization.
* **Interactive Dashboard:** A user-friendly web interface to test the model with custom inputs.
* **Real-time Analysis:** Instant prediction of fraud probability with visual risk assessments.
* **Data Inspector:** Transparent view of the feature vectors (PCA components) being fed into the model.
* **Imbalanced Data Handling:** Utilizes Class Weighting strategies to accurately detect rare fraud events.

---

## 🛠️ Tech Stack

* **Language:** Python 3.8.10
* **Machine Learning:** Scikit-Learn (Logistic Regression, StandardScaler)
* **Data Manipulation:** Pandas, NumPy
* **Model Serialization:** Joblib
* **Frontend/UI:** Streamlit
* **Dataset:** Kaggle European Credit Card Fraud Dataset

---

## 📂 Project Structure

```text
fraud_project/
│
├── creditcard.csv        # The dataset (Download from Kaggle - Ignored by Git)
├── train_model.py        # Script to train and save the model
├── app.py                # The Streamlit interactive dashboard
├── fraud_model.pkl       # Saved model artifacts (Ignored by Git)
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation
````

-----

## 🚀 Installation & Setup

### 1\. Clone the Repository

```bash
git clone https://github.com/risheekbajaj/Credit_card_fraud_detection.git
cd Credit_card_fraud_detection
```

### 2\. Create a Virtual Environment (Optional but Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Download the Data

1.  Go to the [Kaggle Dataset Page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2.  Download `archive.zip` and extract `creditcard.csv`.
3.  Place `creditcard.csv` inside the `fraud_project/` folder.

-----

## ⚡ Usage Guide

### Step 1: Train the Model

Before running the app, you must train the model once. This script processes the data, trains the Logistic Regression classifier, and saves the model to `fraud_model.pkl`.

```bash
python train_model.py
```

*Wait for the message: `✅ SUCCESS: Model saved to 'fraud_model.pkl'`*

### Step 2: Run the Dashboard

Launch the interactive web application.

```bash
python -m streamlit run app.py
```

*The app should automatically open in your web browser at `http://localhost:8501`.*

-----

## 🤝 Contributing

Contributions are welcome\! Please fork the repository and submit a pull request.

## 📜 License

This project is licensed under the MIT License.

```