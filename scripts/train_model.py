# scripts/train_model.py

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import clean_text
from src.feature_extraction import get_tfidf_features
from src.model import train_logistic_regression, save_model

# Wczytanie danych
df = pd.read_csv('../data/raw/sentiment140.csv', encoding='latin-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Przekształcenie etykiet
def label_sentiment(target):
    if target == 0:
        return 'negative'
    elif target == 2:
        return 'neutral'
    elif target == 4:
        return 'positive'

df['sentiment'] = df['target'].apply(label_sentiment)

# Czyszczenie tekstu
df['clean_text'] = df['text'].apply(clean_text)

# Usunięcie niepotrzebnych kolumn
df = df[['sentiment', 'clean_text']].dropna()

# Podział na zbiory treningowy i testowy
X = df['clean_text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ekstrakcja cech
X_train_tfidf, X_test_tfidf, vectorizer = get_tfidf_features(X_train, X_test)

# Trenowanie modelu
model = train_logistic_regression(X_train_tfidf, y_train)

# Zapisanie modelu
save_model(model, '../models/logistic_regression.pkl')

# Zapisanie wektoryzatora
import joblib
joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')

print("Model i wektoryzator zostały zapisane.")
