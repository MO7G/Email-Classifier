import imaplib
import imaplib
import base64
import os
import email
import streamlit as st
import streamlit as st
import time
from nltk.tokenize import word_tokenize
import pandas as pd
from io import StringIO
from html.parser import HTMLParser
from email.header import decode_header, make_header
from bs4 import BeautifulSoup
import re
import os

import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import nltk
from nltk.tokenize import word_tokenize
from termcolor import colored
from nltk.corpus import stopwords
import nltk
from scipy.sparse import csr_matrix, vstack
nltk.download('stopwords')
import numpy as np

import email.header
import work as wc

def extract_spam_emails(username, password,flag, server="imap.gmail.com"):
    # Connect to the IMAP server
    print(username)
    print(password)
    print(server)
    print(flag)
    mail = imaplib.IMAP4_SSL(server)


    # Login to the email account
    login_result, _ = mail.login(username, password)
    if login_result != 'OK':
         st.error("Login failed. Please check your credentials.")
    else:
        st.success("successful login");
    # Select the spam folder
    mail.select("[Gmail]/Spam")

    # Search for all spam emails
    _, message_ids = mail.search(None, "ALL")

    spam_emails = []  # Array to store spam mail bodies
    counter = 0;
    # Iterate over each email message
    for message_id in message_ids[0].split():
        if (counter == flag):
            break;
        _, data = mail.fetch(message_id, "(RFC822)")  # Fetch the email message
        # Parse the message data
        for response_part in data:
            if(counter == flag):
                break;
            if isinstance(response_part, tuple):
                if (counter == flag):
                    break;
                email_body = response_part[1]  # Extract the email body
                spam_emails.append(email_body)  # Add the body to the array
                print(email_body)
                okay =st.success(f"⏳ {counter} extracted")
                st.balloons()
                del okay;
                counter=counter+1
    # Close the connection
    mail.close()
    mail.logout()
    clean_array = []
    for i in spam_emails:
        soup = BeautifulSoup(i, features="html.parser")
        cleaner = soup.get_text();
        clean_array.append(wc.clean(cleaner))
    temp = np.array(clean_array)
    pd.DataFrame(temp).to_csv("file.csv")
    st.success("spam mails are extracted ");




