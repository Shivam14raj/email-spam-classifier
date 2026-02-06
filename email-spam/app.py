import streamlit as st
import pickle

# 
import streamlit as st
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join(BASE_DIR, "spam_model.pkl"), "rb") as f:
    model = pickle.load(f)





# 

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("spam_model.pkl", "rb"))

st.title("📧 Spam Email Classifier")

email_text = st.text_area("Enter the email content")

if st.button("Check Spam"):
    if email_text.strip() == "":
        st.warning("Please enter some text")
    else:
        email_vector = vectorizer.transform([email_text])
        prediction = model.predict(email_vector)

        if prediction[0] == 0:
            st.error("🚨 This email is SPAM")
        else:
            st.success("✅ This email is NOT spam")
