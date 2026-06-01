import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]

print("Original Columns: ")
print(df.columns)

print("\nFirst 5 rows of dataset: ")
print(df.head())




# Rename Columns for better readability 
df = df.rename(columns={'v1':'label', 'v2':'message'})

print("\nAfter Renaming Columns: ")
print(df.head())



# Explore Dataset (EDA)
print("\nShape of dataset: ")
print(df.shape)

print("\nSpam vs Ham Count: ")
print(df['label'].value_counts())

print("\nNull Values: ")
print(df.isnull().sum())



# Data Preprocessing
df['label_num'] = df['label'].map({'ham':0,'spam':1})


# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'],
    test_size=0.2, random_state=42
)

print("\nTraining size: ", X_train.shape)
print("\nTesting Size: ", X_test.shape)



# Feature Extraction
vectorizer = CountVectorizer(
    stop_words='english',
    max_features=3000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.fit_transform(X_test)

print(f"\nVocabulary Size: {len(vectorizer.vocabulary_)}")



# Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)




# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report: ")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))



# Confusion Matrix(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title('confusion_matrix - Spam Email Detection')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()



# Top Spam Words Chart
feature_names = vectorizer.get_feature_names_out()
log_prob_spam = model.feature_log_prob_[1]

top_indices = log_prob_spam.argsort()[-15:]
top_words = [feature_names[i] for i in top_indices]
top_scores = [log_prob_spam[i] for i in top_indices]

plt.figure(figsize=(7,5))
plt.barh(top_words, top_scores, color='red')
plt.title('Top 15 Spam Words')
plt.ylabel('Log probability')
plt.tight_layout()
plt.savefig('top_spam_words.png')
plt.show()