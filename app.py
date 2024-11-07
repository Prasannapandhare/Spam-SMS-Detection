from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Load the model and vectorizer
with open('models/spam_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Initialize NLP tools
wnl = WordNetLemmatizer()

def preprocess_message(message):
    # Clean and preprocess the message
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=message)
    message = message.lower()
    message_words = message.split()
    message_words = [word for word in message_words if word not in set(stopwords.words('english'))]
    final_message = [wnl.lemmatize(word) for word in message_words]
    return ' '.join(final_message)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        sample_message = request.form['message']
        processed_message = preprocess_message(sample_message)
        message_vector = tfidf.transform([processed_message]).toarray()
        prediction = model.predict(message_vector)
        prediction = 'Gotcha! This is a SPAM message.' if prediction[0] == 1 else 'This is a HAM (normal) message.'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
