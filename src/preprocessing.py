import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Process the text (lowercase everything, remove punctuation)
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Vectorize the text (remove stop words, etc.)
def vectorize_text(train_texts, test_texts):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer
