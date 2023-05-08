#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import nltk
import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud


# In[81]:


#Loading CSV file
df = pd.read_csv('Corona_NLP_test.csv')


# In[82]:


df.head()


# In[83]:


#Checking if there is any null values
df.isnull().sum()


# In[84]:


df.columns


# In[85]:


#Taking the columns which is used for analysis
tweet = pd.read_csv('Corona_NLP_test.csv', usecols= ["OriginalTweet"])


# In[86]:


tweet.head()


# QUESTION A. Convert the text corpus into tokens.

# In[87]:


nltk.download('punkt')


# In[91]:


tokenized_tweet = df['OriginalTweet'].apply(lambda x: word_tokenize(x.lower()))
tokenized_tweet[0]


# QUESTION B. Performing stop word removal

# In[89]:


stop_words = set(stopwords.words('english'))
to_remove = ['•', '!', '"', '#', '”', '“', '$', '%', '&', "'", '–', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '…', 'https']
stop_words.update(to_remove)


filtered_text = [[word for word in tweet if not word in stop_words] for tweet in tokenized_tweet]
filtered_text[0]


# QUESTION 3. Counting Word frequencies

# In[92]:


word_freq = {}
for tokenized_tweet in filtered_text:
    for word in tokenized_tweet:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
list(word_freq.items())[:15]


# QUESTION D. Creating word clouds

# In[93]:


def plot_cloud(wordcloud):  
    # Set figure size  
    plt.figure(figsize=(40, 30))  
    # Display image  
    plt.imshow(wordcloud)   
    # No axis details  
    plt.axis("off")
    
wordcloud = WordCloud(width=800, height=800, background_color='#008B8B', colormap='inferno', max_words=50, stopwords = stop_words).generate_from_frequencies(word_freq)
plot_cloud(wordcloud)  


# In[ ]:




