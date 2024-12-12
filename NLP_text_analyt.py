#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#TEXT ANALYTICS USING NLP


# In[1]:


import os
os.chdir('C:\\Users\\ASUS\\desktop\\DL')


# In[3]:


import pandas as pd

#read the data into a pandas dataframe
df = pd.read_csv("Emotion_classify_Data.csv")
print(df.shape)
df.head(5)


# In[5]:


df['label_Emotion'] = df['Emotion'].map({
    'fear':-1,
    'anger':0,
    'joy':1
})
df.head()


# In[7]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.Comment, 
    df.label_Emotion, 
    test_size=0.2, # 20% samples will go to test dataset
    random_state=2022,
    stratify=df.label_Emotion
)


# In[8]:


print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

md = Pipeline([
    ('vectorizer_tri_grams', CountVectorizer(ngram_range = (3, 3))),                       #using the ngram_range parameter 
    ('random_forest', (RandomForestClassifier()))         
])

#2. fit with X_train and y_train
md.fit(X_train, y_train)


#3. get the predictions for X_test and store it in y_pred
y_pred = md.predict(X_test)


#4. print the classfication report
print(classification_report(y_test, y_pred))


# In[ ]:


#Using random forest classoifier with ngram range of 3,3 we do not get a good performing model. 
#We will try NB with different ngram range to optimise the efficiency oh the model


# In[15]:


from sklearn.naive_bayes import MultinomialNB
md = Pipeline([
    ('vectorizer_tri_grams', CountVectorizer(ngram_range = (1, 2))),                       #using the ngram_range parameter 
    ('Naive_Bayes', (MultinomialNB()))         
])

#2. fit with X_train and y_train
md.fit(X_train, y_train)


#3. get the predictions for X_test and store it in y_pred
y_pred = md.predict(X_test)


#4. print the classfication report
print(classification_report(y_test, y_pred))


# In[ ]:


#Here accuracy improved significantly


# In[17]:


md = Pipeline([
    ('vectorizer_tri_grams', CountVectorizer(ngram_range = (1, 2))),                       #using the ngram_range parameter 
    ('Naive_Bayes', (RandomForestClassifier()))         
])

#2. fit with X_train and y_train
md.fit(X_train, y_train)


#3. get the predictions for X_test and store it in y_pred
y_pred = md.predict(X_test)


#4. print the classfication report
print(classification_report(y_test, y_pred))


# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer

#1. create a pipeline object
md = Pipeline([
     ('vectorizer_tfidf',TfidfVectorizer()),        #using the ngram_range parameter 
     ('Random Forest', RandomForestClassifier())         
])

#2. fit with X_train and y_train
md.fit(X_train, y_train)


#3. get the predictions for X_test and store it in y_pred
y_pred = md.predict(X_test)


#4. print the classfication report
print(classification_report(y_test, y_pred))


# In[ ]:


#HERE WE ARE DOING PREPROCESSING ON THE COMMENTS TEXT TO REMOVE STOP WORDS, PUNCTUATUATIONS AND CONVERT THE WORDS INTO THEIR BASE WORDS OR LEMMA.
#WE WILL SE A  SIGNIFICANT IMPROVEMNET IN THE MODEL PERFORMANCE BY PREPROCESSING THE TEXT.


# In[ ]:


#WE HAVE CREATED A NEW COLUMN IN THE EXISTING DF WITH THE NAME preprocessed_comment.


# In[21]:


import spacy

# load english language model and create nlp object from it
nlp = spacy.load("en_core_web_sm") 


#use this utility function to get the preprocessed text data
def preprocess(text):
    # remove stop words and lemmatize the text
    doc = nlp(text)
    lemat_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        lemat_tokens.append(token.lemma_)
    
    return " ".join(lemat_tokens) 


# In[23]:


df["preprocessed_comment"] = df['Comment'].apply(preprocess)


# In[24]:


df.head()


# In[29]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.preprocessed_comment, 
    df.label_Emotion, 
    test_size=0.2,
    random_state=2022,
    stratify=df.label_Emotion
)


# In[31]:


md = Pipeline([
    ('vectorizer_tri_grams', CountVectorizer(ngram_range = (1, 2))),                       #using the ngram_range parameter 
    ('Naive_Bayes', (RandomForestClassifier()))         
])

#2. fit with X_train and y_train
md.fit(X_train, y_train)


#3. get the predictions for X_test and store it in y_pred
y_pred = md.predict(X_test)


#4. print the classfication report
print(classification_report(y_test, y_pred))


# In[ ]:


#THE ABOVE MODEL SEEMS TO PERFORM REALLY WELL ON THE GIVEN TEXT AFTER PREPROCESSING WITH EXCELLENT ACCURACY, F1 SCORE AND RECALL.
#IT SIGNIFIES THE IMPORTANCE OF DATA PREPROCESSING ESPECIALLY IN TEXT ANALYTICS WHICH COULD SAVE TIME AND SPACE AND ULTIMATELY THE OPERATIONAL COST.


# In[32]:


from sklearn.feature_extraction.text import TfidfVectorizer

#1. create a pipeline object
md = Pipeline([
     ('vectorizer_tfidf',TfidfVectorizer()),        #using the ngram_range parameter 
     ('Random Forest', RandomForestClassifier())         
])

#2. fit with X_train and y_train
md.fit(X_train, y_train)


#3. get the predictions for X_test and store it in y_pred
y_pred = clf.predict(X_test)


#4. print the classfication report
print(classification_report(y_test, y_pred))


# In[ ]:




