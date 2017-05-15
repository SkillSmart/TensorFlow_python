
# coding: utf-8

# # Text Data Preprocessing Routine
# 
# [Example Video](https://www.youtube.com/watch?v=YFxVHD2TNII&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=49)<br>
# Author: Frank Fichtenmueller<br>
# Start Date: 11:30<br>
# End Time: 12:45<br>
# <hr>
# 
# Goal:<br> Implement the Core Routines to create a Bag of words representation of a positive and negative Sentiment Dataset
# 
# Applied Methods:<br> Implemted Functions to take a raw textsource, extract the words, apply grammatical word stemming, remove unwanted stopwords(by only count), count the number of occurances, and return a dictionary object 

# In[30]:

import nltk
import pandas as pd
import numpy as np
import random
import pickle
from collections import Counter


# bFirst we will create a lexicon from the Datacorpus. In this case we are dealing with only two files. Positive and negative Sentiment Data

# In[31]:

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Tokenizer splits string into words

# In[32]:

s = 'i pulled the chair up to the table'
word_tokenize(s)


# In[33]:

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000


# In[34]:

lemmatizer.lemmatize(s)


# In[35]:

lemmatizer.lemmatize('liked')


# ### First we set up a lexicon in the dataset

# In[36]:

# A basic implementation to create a lexicon
def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos,neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
                
    # Now we stemm the words to get rid of too many synonyms in the dataset
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    # This creates a dict of counts for each object in lexicon: {'the':55321, 'after':..}
    w_counts = Counter(lexicon)
    
    l2 = []
    # We filter for uncommon words
    for w in w_counts:
        if 50 < w_counts[w] < 1000:
            l2.append(w)
    
    return l2


# In[49]:

def sample_handling(sample, lexicon, classification):
    featureset = []
    
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
    
    return featureset


# In[52]:

def create_features_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling(pos, lexicon, [1,0])
    features += sample_handling(neg, lexicon, [0,1])
    random.shuffle(features)
    
    features = np.array(features)
    testing_size = int(test_size*len(features))
    
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    
    return train_x, train_y, test_x, test_y


# In[53]:

train_x, train_y, test_x, test_y = create_features_sets_and_labels('Data_raw/pos.txt', 'Data_raw/neg.txt')
with open('sentiment_set.pickle', 'wb') as f:
    pickle.dump([train_x, train_y, test_x, test_y], f)


# In[ ]:



