# src/data_preprocessing.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words('english'))  # Zmień na 'polish' dla polskich danych

def clean_text(text):
    # Usuwanie linków
    text = re.sub(r'http\S+', '', text)
    # Usuwanie znaków specjalnych i cyfr
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenizacja
    tokens = nltk.word_tokenize(text.lower())
    # Usuwanie stop-words i lematyzacja
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop]
    return ' '.join(tokens)