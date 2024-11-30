import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Step 1: Load the datasets
train_data = pd.read_excel("/content/Train_Task_B.xlsx")
val_data = pd.read_excel("/content/Val_Task_B.xlsx")
test_data = pd.read_excel("/content/Test_Task_B.xlsx")

# Step 2: Text Cleaning Function
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        return text
    return ""

# Step 3: Apply Text Cleaning to the Datasets
train_data['Cleaned_Tweet'] = train_data['Tweet'].apply(clean_text)
val_data['Cleaned_Tweet'] = val_data['Tweet'].apply(clean_text)
test_data['Cleaned_Tweet'] = test_data['Tweet'].apply(clean_text)

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_data['Cleaned_Tweet'])
X_val = vectorizer.transform(val_data['Cleaned_Tweet'])
X_test = vectorizer.transform(test_data['Cleaned_Tweet'])

# Step 5: Feature Engineering - Text Length (Optional but may help with performance)
train_data['Text_Length'] = train_data['Cleaned_Tweet'].apply(len)
val_data['Text_Length'] = val_data['Cleaned_Tweet'].apply(len)
test_data['Text_Length'] = test_data['Cleaned_Tweet'].apply(len)

# Combine TF-IDF features with text length feature
X_train_combined = np.hstack((X_train.toarray(), train_data[['Text_Length']].values))
X_val_combined = np.hstack((X_val.toarray(), val_data[['Text_Length']].values))
X_test_combined = np.hstack((X_test.toarray(), test_data[['Text_Length']].values))

# Step 6: Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)

# Targets for Hate Speech and Fake News
y_train_hate = train_data['Hate']
y_val_hate = val_data['Hate']
y_train_fake = train_data['Fake']
y_val_fake = val_data['Fake']

# Apply SMOTE to balance both classes (Hate and Fake)
X_train_smote_hate, y_train_smote_hate = smote.fit_resample(X_train_combined, y_train_hate)
X_train_smote_fake, y_train_smote_fake = smote.fit_resample(X_train_combined, y_train_fake)

# Step 7: Train Random Forest Model for Hate Speech Detection
rf_model_hate = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_hate.fit(X_train_smote_hate, y_train_smote_hate)

# Step 8: Train Random Forest Model for Fake News Detection
rf_model_fake = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_fake.fit(X_train_smote_fake, y_train_smote_fake)

# Step 9: Predict on Validation Set for Hate Speech
y_val_pred_hate = rf_model_hate.predict(X_val_combined)

# Step 10: Predict on Validation Set for Fake News
y_val_pred_fake = rf_model_fake.predict(X_val_combined)

# Step 11: Print Classification Report for Validation
print("Hate Speech Detection Accuracy:", accuracy_score(y_val_hate, y_val_pred_hate))
print("Classification Report (Hate):\n", classification_report(y_val_hate, y_val_pred_hate))

print("Fake News Detection Accuracy:", accuracy_score(y_val_fake, y_val_pred_fake))
print("Classification Report (Fake):\n", classification_report(y_val_fake, y_val_pred_fake))

# Step 12: Make Predictions on Test Data
test_predictions_hate = rf_model_hate.predict(X_test_combined)
test_predictions_fake = rf_model_fake.predict(X_test_combined)

# Step 13: Prepare the Submission Dataframe
submission = test_data[['Id']].copy()
submission['Hate'] = test_predictions_hate
submission['Fake'] = test_predictions_fake

# Ensure "Target" and "Severity" columns are filled with "N/A"
submission['Target'] = 'N/A'
submission['Severity'] = 'N/A'

# Step 14: Save the Predictions to CSV
submission.to_csv("/content/Final_Submission_B.csv", index=False)
print("Predictions saved to 'Final_Submission_B.csv'.")
