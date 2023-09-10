import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
Here's an improved version of the Python program:

```python


class FraudDetectionSystem:
    def __init__(self):
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.classifier = None

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)

    def preprocess_data(self):
        # Drop irrelevant columns
        self.data.drop(['transaction_id', 'user_id'], axis=1, inplace=True)

        # Convert categorical variables to numerical using one-hot encoding
        self.data = pd.get_dummies(self.data, drop_first=True)

        # Standardize numerical features
        scaler = StandardScaler()
        self.data[['purchase_amount', 'time_spent']] = scaler.fit_transform(
            self.data[['purchase_amount', 'time_spent']])

    def train_test_split(self):
        X = self.data.drop('fraudulent', axis=1)
        y = self.data['fraudulent']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

    def train(self):
        # Initialize the Isolation Forest classifier
        self.classifier = IsolationForest(
            n_estimators=100, contamination='auto', random_state=42)

        # Train the classifier
        self.classifier.fit(self.X_train)

    def evaluate(self):
        # Predict the labels for the test set
        y_pred = self.classifier.predict(self.X_test)

        # Convert -1 (outliers) to 1 and 1 (inliers) to 0 for evaluation metrics
        y_pred[y_pred == -1] = 1
        y_pred[y_pred == 1] = 0

        # Calculate accuracy, precision, recall, and F1-score
        accuracy = np.mean(y_pred == self.y_test)
        tn, fp, fn, tp = confusion_matrix(
            self.y_test, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)

        # Print evaluation metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")

        # Plot confusion matrix
        cm = confusion_matrix(self.y_test, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

    def detect_fraud(self, new_data):
        # Preprocess new data
        new_data = pd.get_dummies(new_data, drop_first=True)
        new_data[['purchase_amount', 'time_spent']] = StandardScaler(
        ).fit_transform(new_data[['purchase_amount', 'time_spent']])

        # Predict the labels for new data
        y_pred = self.classifier.predict(new_data)

        # Convert -1 (outliers) to 1 and 1 (inliers) to 0
        y_pred[y_pred == -1] = 1
        y_pred[y_pred == 1] = 0

        return y_pred


# Instantiate the fraud detection system
fds = FraudDetectionSystem()

# Load data
fds.load_data('transactions.csv')

# Preprocess data
fds.preprocess_data()

# Split data into train and test sets
fds.train_test_split()

# Train the model
fds.train()

# Evaluate the model
fds.evaluate()

# Example usage: Detect fraud for new transactions
new_transactions = pd.DataFrame([
    {'purchase_amount': -0.8, 'time_spent': -1.2, 'payment_method_credit_card': 1,
        'payment_method_paypal': 0, 'payment_method_upi': 0},
    {'purchase_amount': 1.5, 'time_spent': 0.7, 'payment_method_credit_card': 0,
        'payment_method_paypal': 1, 'payment_method_upi': 0},
    {'purchase_amount': 0.3, 'time_spent': 1.0, 'payment_method_credit_card': 0,
        'payment_method_paypal': 0, 'payment_method_upi': 1}
])

fraud_labels = fds.detect_fraud(new_transactions)
print(fraud_labels)
```

Improvements made:
1. Used `inplace = True` when dropping irrelevant columns to modify the DataFrame in -place instead of returning a new DataFrame.
2. Created a separate method for each step in the fraud detection process to improve readability and maintainability.
3. Moved the initialization of `scaler` inside the `preprocess_data` method to ensure a new instance is created for each preprocessing call.
4. Used `np.mean` instead of manually calculating the accuracy for evaluation.
5. Assigned the confusion matrix to `cm` variable for readability and reused it for plotting the confusion matrix.
6. Added comments to explain the purpose of each method and improve code documentation.
