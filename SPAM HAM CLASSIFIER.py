import pandas as pd
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#print(" Loading dataset...")


df = pd.read_csv("spam_ham_dataset.csv", engine='python')
df = df[['label_num', 'text']]
df.rename(columns={'label_num': 'label'}, inplace=True)

print(" Dataset loaded successfully!")
print(f"Total samples: {len(df)}")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

#print("Cleaning text data...")
df['text'] = df['text'].apply(clean_text)

#print("Training model...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    max_df=0.95,
    min_df=5
)

X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print(" Model ready for predictions! ✅ ")
print("📧 Type an email message to classify (or type 'exit' to quit):")

# Prediction Loop
while True:
    user_input = input("\nYour email: ")
    if user_input.lower() == 'exit':
        print("👋 Exiting classifier. Goodbye!")
        break
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    print("📢 Prediction:", "SPAM" if prediction == 1 else "HAM")
