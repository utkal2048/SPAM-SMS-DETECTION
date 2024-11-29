import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources if not already downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load model and vectorizer
try:
    tk = pickle.load(open("vectorizer.pkl", 'rb'))
    model = pickle.load(open("model.pkl", 'rb'))
except Exception as e:
    st.error("Error loading model or vectorizer: {}".format(e))

st.title("SMS Spam Detection Model")


input_sms = st.text_input("Enter the SMS")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.error("Please enter a valid SMS message.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tk.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
