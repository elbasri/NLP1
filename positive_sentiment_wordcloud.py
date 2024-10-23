import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
import pickle

# Load the dataset
dataset = pd.read_csv('data/sentiment_analysis.csv')

# Select relevant columns
dataset = dataset[['text', 'sentiment']]

# Convert text to lowercase
dataset['text'] = dataset['text'].str.lower()

# Download stopwords and load English stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Function to remove stopwords from text
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Apply stopwords removal
dataset['text'] = dataset['text'].apply(remove_stopwords)

# Filter positive sentiment texts for word cloud generation
positive_texts = dataset[dataset['sentiment'] == 'positive']

# Generate word cloud for positive texts
text = positive_texts.text
wordcloud = WordCloud(
    width=2000,
    height=1000,
    background_color='black',
    stopwords=STOPWORDS
).generate(str(text))

# Plot the word cloud
fig = plt.figure(
    figsize=(40, 30),
    facecolor='k',
    edgecolor='k'
)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

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

# Train a Naive Bayes classifier on CountVectorizer features
nb_count = MultinomialNB()
nb_count.fit(X_train_count, y_train)

# Predict using the CountVectorizer features
y_pred_count = nb_count.predict(X_test_count)
print("CountVectorizer - Accuracy: ", accuracy_score(y_test, y_pred_count))
print(classification_report(y_test, y_pred_count))

# Train a Naive Bayes classifier on TfidfVectorizer features
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)

# Predict using the TfidfVectorizer features
y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)
print("TfidfVectorizer - Accuracy: ", accuracy_score(y_test, y_pred_tfidf))
print(classification_report(y_test, y_pred_tfidf))

# Save the CountVectorizer model and vectorizer
with open('models/nb_count_model.pkl', 'wb') as model_file:
    pickle.dump(nb_count, model_file)

with open('models/count_vectorizer.pkl', 'wb') as vect_file:
    pickle.dump(count_vect, vect_file)

# Save the TfidfVectorizer model and vectorizer
with open('models/nb_tfidf_model.pkl', 'wb') as model_file:
    pickle.dump(nb_tfidf, model_file)

with open('models/tfidf_vectorizer.pkl', 'wb') as vect_file:
    pickle.dump(tfidf_vect, vect_file)
