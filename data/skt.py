import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Sample data
# Assuming you have a dataset where 'text' is the column with text and 'sentiment' has labels (positive/negative)
dataset = pd.read_csv('/content/sentiment_analysis.csv')
dataset['text'] = dataset['text'].str.lower()

# Function to remove stopwords
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Apply stopwords removal
dataset['text'] = dataset['text'].apply(remove_stopwords)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['sentiment'], test_size=0.2, random_state=42)

# 1. CountVectorizer
count_vect = CountVectorizer(max_features=1000)
X_train_count = count_vect.fit_transform(X_train)
X_test_count = count_vect.transform(X_test)

# 2. TfidfVectorizer
tfidf_vect = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vect.fit_transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)

# Train a classifier (Naive Bayes as an example) on the CountVectorizer features
nb_count = MultinomialNB()
nb_count.fit(X_train_count, y_train)

# Predict using the CountVectorizer features
y_pred_count = nb_count.predict(X_test_count)
print("CountVectorizer - Accuracy: ", accuracy_score(y_test, y_pred_count))
print(classification_report(y_test, y_pred_count))

# Train a classifier (Naive Bayes) on the TfidfVectorizer features
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)

# Predict using the TfidfVectorizer features
y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)
print("TfidfVectorizer - Accuracy: ", accuracy_score(y_test, y_pred_tfidf))
print(classification_report(y_test, y_pred_tfidf))
