import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import joblib

# Load the dataset
file_path = "/content/drive/My Drive/dataset/data.csv"  # Replace with the correct file path
df = pd.read_csv(file_path)

# Display the first few rows
print("Dataset Preview:")
print(df.head())

# Encode the categorical 'crop' column using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['crop'], drop_first=True)

# Define features (X) and target (y)
X = df_encoded.drop(columns=['pump'])
y = df_encoded['pump']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
joblib.dump(model, 'irrigation_model_with_crop.pkl')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['OFF', 'ON'], yticklabels=['OFF', 'ON'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['OFF', 'ON']))

# Feature Importance
feature_importance = model.feature_importances_
for feature, importance in zip(X.columns, feature_importance):
    print(f'Feature: {feature}, Importance: {importance:.4f}')
    
# Plot feature importance
plt.figure(figsize=(8, 6))
plt.bar(X.columns, feature_importance)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()

