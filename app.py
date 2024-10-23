import tkinter as tk
from tkinter import messagebox, scrolledtext
import pickle
from nltk.corpus import stopwords
import nltk

# Load the saved models and vectorizers
with open('models/nb_count_model.pkl', 'rb') as model_file:
    nb_count_model = pickle.load(model_file)

with open('models/count_vectorizer.pkl', 'rb') as vect_file:
    count_vectorizer = pickle.load(vect_file)

with open('models/nb_tfidf_model.pkl', 'rb') as model_file:
    nb_tfidf_model = pickle.load(model_file)

with open('models/tfidf_vectorizer.pkl', 'rb') as vect_file:
    tfidf_vectorizer = pickle.load(vect_file)

# Download stopwords if necessary
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to remove stopwords from input text
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Function to predict sentiment based on user input
def predict_sentiment():
    # Get the text input from the user
    user_input = input_text.get("1.0", tk.END).strip()
    
    if not user_input:
        messagebox.showerror("Input Error", "Please enter some text to analyze.")
        return
    
    # Preprocess the text (lowercasing and removing stopwords)
    processed_text = remove_stopwords(user_input.lower())
    
    # Predict with CountVectorizer model
    vectorized_text_count = count_vectorizer.transform([processed_text])
    sentiment_count = nb_count_model.predict(vectorized_text_count)
    
    # Predict with TfidfVectorizer model
    vectorized_text_tfidf = tfidf_vectorizer.transform([processed_text])
    sentiment_tfidf = nb_tfidf_model.predict(vectorized_text_tfidf)
    
    # Display the result in the interface
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, f"CountVectorizer Prediction: {sentiment_count[0]}\n")
    result_text.insert(tk.END, f"TfidfVectorizer Prediction: {sentiment_tfidf[0]}\n")

# Create the main Tkinter window
root = tk.Tk()
root.title("Application d'analyse des sentiments")

# Configure the window size and background
root.geometry("500x400")
root.configure(bg="#f0f0f0")

# Define the interface layout using a grid
frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(padx=20, pady=20)

# Set custom font and colors
font_label = ("Arial", 12, "bold")
font_text = ("Arial", 10)

# Label for input field
input_label = tk.Label(frame, text="Entrez du texte pour l'analyse des sentiments:", bg="#f0f0f0", font=font_label)
input_label.grid(row=0, column=0, sticky="w", pady=5)

# Scrollable Textbox for user input
input_text = scrolledtext.ScrolledText(frame, height=5, width=50, wrap=tk.WORD, font=font_text)
input_text.grid(row=1, column=0, padx=5, pady=5)

# Button to trigger sentiment prediction with a better size and styling
predict_button = tk.Button(frame, text="Prédire le sentiment", command=predict_sentiment, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), width=20, height=2)
predict_button.grid(row=2, column=0, pady=10)

# Label for result
result_label = tk.Label(frame, text="Résultats de prédiction:", bg="#f0f0f0", font=font_label)
result_label.grid(row=3, column=0, sticky="w", pady=5)

# Scrollable Textbox to display the results
result_text = scrolledtext.ScrolledText(frame, height=5, width=50, wrap=tk.WORD, font=font_text)
result_text.grid(row=4, column=0, padx=5, pady=5)

# Run the Tkinter event loop
root.mainloop()
