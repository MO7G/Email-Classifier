import streamlit as st
import pandas as pd
import numpy as np
import work as wr
import extractor as ex
base="dark"
primaryColor="purple"
username = "mohd.alhajj1913@gmail.com"
password = "vmkjisitoxokmyeg"
st.title("Email Classifier")

    # Add your GUI components here
st.write("Welcome to this project !")
st.write("Email Classification with Emotion Detection and Topic Tagging project is designed to analyze and categorize emails based on their content. The project utilizes various natural language processing (NLP) techniques to classify emails as spam or not spam, determine the emotions expressed in the text, and extract topic tags to provide a comprehensive understanding of the email content.")
email = st.text_input("Email:")
password = st.text_input("Password")
NumberOfEmails = st.slider("Select Number of emails to extract:", min_value=1, max_value=50)
okay=0;
col3,col4,col5 = st.columns([1,2,5])
with col3:
     process= st.button("login");
with col4:
     global_cleaning= st.button("clean")
     if(global_cleaning ==True):
       wr.global_work();


show_dataFrame = st.button("Results")
if(show_dataFrame == True):
    sample = pd.read_csv("final.csv")
    st.write(sample)


if process == True:
    ex.extract_spam_emails(email, password,NumberOfEmails);




typeOfOperation = ['Spam and Ham Checker', 'Emotion Checker', 'Tag Chcker']

# Create the dropdown in Streamlit
selected_option = st.selectbox('Select an option', typeOfOperation)

# Display the selected option
st.write('You selected:', selected_option)
dataTest = st.text_area("Manual Checker Enter your email and chose what type of checking do you want to perform");

col1,col2 = st.columns([0.4,3])
with col1:
     show= st.button("show");
with col2:
     close= st.button("close")
if show == True:
    if selected_option == typeOfOperation[0]:
       result= wr.predictSpamOrHam(dataTest);
       if result == "Spam":
          st.error(result)
       else:
          st.success(result)
    elif selected_option == typeOfOperation[1]:
       result= wr.predictEmotion(dataTest);
       st.success(result)

    elif selected_option == typeOfOperation[2]:
       result= wr.predictTag([dataTest]);
       st.success(result)

if close == True:
    show = False;



