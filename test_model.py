from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

# Load test dataset
test_data = pd.read_csv("mitbih_test.csv", header=None)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].astype(int).values

# Normalize and reshape
X_test = X_test / np.max(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# One-hot encode labels
encoder = LabelBinarizer()
encoder.fit([0, 1, 2, 3, 4])
y_test_encoded = encoder.transform(y_test)

# Load the trained model
model = load_model("ecg_cnn_final.h5")

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict and report
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test_encoded, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
