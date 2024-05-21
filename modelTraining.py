import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Define the root data directory
data_root = 'data'

# Initialize lists to hold the data and labels
s21_data = []
labels = []

# Walk through the directory structure
for root, dirs, files in os.walk(data_root):
    for file in files:
        if file.endswith('.npz'):
            # Extract the label from the directory name
            label = os.path.basename(root)
            # Load the data from the .npz file
            file_path = os.path.join(root, file)
            data = np.load(file_path, allow_pickle=True)['data']
            for entry in data:
                s21_data.append(entry['s21'])
                labels.append(label)

# Convert the lists to numpy arrays
s21_data = np.array(s21_data)
labels = np.array(labels)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(s21_data, labels, test_size=0.2, random_state=42)

# Create a scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

# Save the trained model and scaler to files
model_filename = 's21_svm_model.joblib'
scaler_filename = 'scaler_s21.pkl'
joblib.dump(model, model_filename)
joblib.dump(scaler, scaler_filename)
print(f"Model saved to {model_filename}")
print(f"Scaler saved to {scaler_filename}")

# Predict the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output the results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
