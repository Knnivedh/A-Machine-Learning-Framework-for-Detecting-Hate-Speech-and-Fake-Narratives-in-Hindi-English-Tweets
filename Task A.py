# Import required libraries
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import tensorflow as tf

# Step 1: Load the datasets
train_task_a = pd.read_excel("/content/Train_Task_A.xlsx")
val_task_a = pd.read_excel("/content/Val_Task_A.xlsx")
test_task_a = pd.read_excel("/content/Test_Task_A.xlsx")

# Step 2: Text cleaning function
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        return text
    return ""

# Apply text cleaning to the data
train_task_a['Cleaned_Tweet'] = train_task_a['Tweet'].apply(clean_text)
val_task_a['Cleaned_Tweet'] = val_task_a['Tweet'].apply(clean_text)
test_task_a['Cleaned_Tweet'] = test_task_a['Tweet'].apply(clean_text)

# Step 3: TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train = tfidf.fit_transform(train_task_a['Cleaned_Tweet'])
X_val = tfidf.transform(val_task_a['Cleaned_Tweet'])
X_test = tfidf.transform(test_task_a['Cleaned_Tweet'])

# Step 4: Feature Engineering (Text Length)
train_task_a['Text_Length'] = train_task_a['Cleaned_Tweet'].apply(len)
val_task_a['Text_Length'] = val_task_a['Cleaned_Tweet'].apply(len)
test_task_a['Text_Length'] = test_task_a['Cleaned_Tweet'].apply(len)

X_train_combined = np.hstack((X_train.toarray(), train_task_a[['Text_Length']].values))
X_val_combined = np.hstack((X_val.toarray(), val_task_a[['Text_Length']].values))
X_test_combined = np.hstack((X_test.toarray(), test_task_a[['Text_Length']].values))

# Targets for individual tasks (Hate, Fake)
y_train_hate = train_task_a['Hate']
y_val_hate = val_task_a['Hate']

# Step 5: Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_combined, y_train_hate)

# Step 6: Train Random Forest Model for Hate Speech Detection
rf_model_hate = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_hate.fit(X_train_smote, y_train_smote)

# Predict on validation set for Hate Speech
y_val_pred_hate = rf_model_hate.predict(X_val_combined)

# Print classification report for Hate Speech Detection
print("Hate Speech Detection Accuracy:", accuracy_score(y_val_hate, y_val_pred_hate))
print("Classification Report (Hate):\n", classification_report(y_val_hate, y_val_pred_hate))

# Step 7: Train Random Forest Model for Fake News Detection
y_train_fake = train_task_a['Fake']
y_val_fake = val_task_a['Fake']

# Handle class imbalance with SMOTE for Fake News
X_train_smote_fake, y_train_smote_fake = smote.fit_resample(X_train_combined, y_train_fake)

rf_model_fake = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_fake.fit(X_train_smote_fake, y_train_smote_fake)

# Predict on validation set for Fake News
y_val_pred_fake = rf_model_fake.predict(X_val_combined)

# Print classification report for Fake News Detection
print("Fake News Detection Accuracy:", accuracy_score(y_val_fake, y_val_pred_fake))
print("Classification Report (Fake):\n", classification_report(y_val_fake, y_val_pred_fake))

# Step 8: Make Predictions on Test Data
test_predictions_hate = rf_model_hate.predict(X_test_combined)
test_predictions_fake = rf_model_fake.predict(X_test_combined)

# Prepare the submission dataframe
submission = test_task_a[['Id']].copy()
submission['Hate'] = test_predictions_hate
submission['Fake'] = test_predictions_fake

# Ensure the "Target" and "Severity" columns are filled with "N/A" (or skipped if not needed)
submission['Target'] = 'N/A'  # If the Target column is missing, fill with 'N/A'
submission['Severity'] = 'N/A'  # If the Severity column is missing, fill with 'N/A'

# Step 9: Save the predictions in the required format
submission.to_csv("Final_Submission.csv", index=False)
print("Predictions saved to 'Final_Submission.csv'.")
