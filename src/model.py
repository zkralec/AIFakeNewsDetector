import pandas as pd

from sklearn.model_selection import train_test_split
from preprocessing import preprocess_text, vectorize_text

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Loading the fake_news CSV file
df = pd.read_csv('env/data/fake_news.csv')

# Split the features and labels
X = df['statement']
y = df['label']

# Split training and testing (80% and 20% respectively)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess the text
X_train_cleaned = X_train.apply(preprocess_text)
X_test_cleaned = X_test.apply(preprocess_text)

# Vectorize the cleaned text
X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train_cleaned, X_test_cleaned)

# Create a training model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_vec, y_train)

# Predict the testing data
y_pred = model.predict(X_test_vec)

# Evaluate the accuracy of training/testing
accuracy = accuracy_score(y_test, y_pred)
print(f'\nTest Accuracy: {accuracy:.4f}')

# Detailed report
print(classification_report(y_test, y_pred))
