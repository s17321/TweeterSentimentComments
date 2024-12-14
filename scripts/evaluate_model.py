# scripts/evaluate_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import clean_text
from src.feature_extraction import get_tfidf_features
from src.model import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

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
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Załadowanie wektoryzatora i modelu
vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')
model = load_model('../models/logistic_regression.pkl')

# Ekstrakcja cech
X_test_tfidf = vectorizer.transform(X_test)

# Predykcje
y_pred = model.predict(X_test_tfidf)

# Ewaluacja
print(classification_report(y_test, y_pred))

# Macierz pomyłek
cm = confusion_matrix(y_test, y_pred, labels=['negative', 'neutral', 'positive'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predykcja')
plt.ylabel('Rzeczywistość')
plt.title('Macierz Pomyłek')
plt.show()
