import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# --- Configuration ---
DATA_FILE = 'creditcard.csv'
MODEL_FILE = 'fraud_model.pkl'

def train():
    print("="*40)
    print("🤖 Model Training Pipeline Started")
    print("="*40)

    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"❌ ERROR: '{DATA_FILE}' not found.")
        print("Please download the dataset from Kaggle and place it in this folder.")
        return

    print("⏳ Loading dataset... (This might take a few seconds)")
    df = pd.read_csv(DATA_FILE)
    
    # 2. Preprocessing
    print("🛠️  Preprocessing data...")
    
    # The dataset has 'Time', 'Amount', and 'V1'...'V28'.
    # V-features are already scaled PCA components. 
    # 'Amount' is not scaled, so we must scale it.
    # 'Time' is generally not useful for this specific model type, so we drop it.
    
    scaler = StandardScaler()
    df['Amount_Scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # Define X (features) and y (target)
    # We drop the original 'Amount' and 'Time'
    X = df.drop(['Time', 'Amount', 'Class'], axis=1)
    y = df['Class']

    # Split: 80% Train, 20% Test
    # Stratify is crucial here to keep the ratio of fraud cases consistent
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Model Training
    print("🚀 Training Logistic Regression Model...")
    # class_weight='balanced' tells the model to pay more attention to the rare fraud cases
    model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
    model.fit(X_train, y_train)
    
    # 4. Evaluation
    print("📊 Evaluating Model...")
    y_pred = model.predict(X_test)
    
    print("\n--- Test Set Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Save Artifacts
    # We save the model, scaler, and column names so the app knows exactly what to do
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }
    
    joblib.dump(model_artifacts, MODEL_FILE)
    print(f"\n✅ SUCCESS: Model saved to '{MODEL_FILE}'")

if __name__ == "__main__":
    train()