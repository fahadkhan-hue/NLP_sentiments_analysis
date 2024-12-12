```python
                                       #TEXT ANALYTICS USING NLP
```


```python
import os
os.chdir('C:\\Users\\ASUS\\desktop\\DL')
```


```python
import pandas as pd

#read the data into a pandas dataframe
df = pd.read_csv("Emotion_classify_Data.csv")
print(df.shape)
df.head(5)
```

    (5937, 2)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Comment</th>
      <th>Emotion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>i seriously hate one subject to death but now ...</td>
      <td>fear</td>
    </tr>
    <tr>
      <th>1</th>
      <td>im so full of life i feel appalled</td>
      <td>anger</td>
    </tr>
    <tr>
      <th>2</th>
      <td>i sit here to write i start to dig out my feel...</td>
      <td>fear</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ive been really angry with r and i feel like a...</td>
      <td>joy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>i feel suspicious if there is no one outside l...</td>
      <td>fear</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['label_Emotion'] = df['Emotion'].map({
    'fear':-1,
    'anger':0,
    'joy':1
})
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Comment</th>
      <th>Emotion</th>
      <th>label_Emotion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>i seriously hate one subject to death but now ...</td>
      <td>fear</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>im so full of life i feel appalled</td>
      <td>anger</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>i sit here to write i start to dig out my feel...</td>
      <td>fear</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ive been really angry with r and i feel like a...</td>
      <td>joy</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>i feel suspicious if there is no one outside l...</td>
      <td>fear</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.Comment, 
    df.label_Emotion, 
    test_size=0.2, # 20% samples will go to test dataset
    random_state=2022,
    stratify=df.label_Emotion
)
```


```python
print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)
```

    Shape of X_train:  (4749,)
    Shape of X_test:  (1188,)
    


```python
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
```

                  precision    recall  f1-score   support
    
              -1       0.40      0.80      0.54       388
               0       0.53      0.32      0.40       400
               1       0.57      0.26      0.36       400
    
        accuracy                           0.45      1188
       macro avg       0.50      0.46      0.43      1188
    weighted avg       0.50      0.45      0.43      1188
    
    


```python
#Using random forest classoifier with ngram range of 3,3 we do not get a good performing model. 
#We will try NB with different ngram range to optimise the efficiency oh the model
```


```python
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
```

                  precision    recall  f1-score   support
    
              -1       0.88      0.89      0.88       388
               0       0.88      0.89      0.88       400
               1       0.89      0.87      0.88       400
    
        accuracy                           0.88      1188
       macro avg       0.88      0.88      0.88      1188
    weighted avg       0.88      0.88      0.88      1188
    
    


```python
#Here accuracy improved significantly
```


```python
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
```

                  precision    recall  f1-score   support
    
              -1       0.97      0.89      0.93       388
               0       0.94      0.88      0.91       400
               1       0.85      0.97      0.91       400
    
        accuracy                           0.91      1188
       macro avg       0.92      0.91      0.91      1188
    weighted avg       0.92      0.91      0.91      1188
    
    


```python
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
```

                  precision    recall  f1-score   support
    
              -1       0.92      0.91      0.91       388
               0       0.92      0.88      0.90       400
               1       0.89      0.94      0.91       400
    
        accuracy                           0.91      1188
       macro avg       0.91      0.91      0.91      1188
    weighted avg       0.91      0.91      0.91      1188
    
    


```python
#HERE WE ARE DOING PREPROCESSING ON THE COMMENTS TEXT TO REMOVE STOP WORDS, PUNCTUATUATIONS AND CONVERT THE WORDS INTO THEIR BASE WORDS OR LEMMA.
#WE WILL SE A  SIGNIFICANT IMPROVEMNET IN THE MODEL PERFORMANCE BY PREPROCESSING THE TEXT.

```


```python
#WE HAVE CREATED A NEW COLUMN IN THE EXISTING DF WITH THE NAME preprocessed_comment.
```


```python
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
```


```python
df["preprocessed_comment"] = df['Comment'].apply(preprocess)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Comment</th>
      <th>Emotion</th>
      <th>label_Emotion</th>
      <th>preprocessed_comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>i seriously hate one subject to death but now ...</td>
      <td>fear</td>
      <td>-1</td>
      <td>seriously hate subject death feel reluctant drop</td>
    </tr>
    <tr>
      <th>1</th>
      <td>im so full of life i feel appalled</td>
      <td>anger</td>
      <td>0</td>
      <td>m life feel appalled</td>
    </tr>
    <tr>
      <th>2</th>
      <td>i sit here to write i start to dig out my feel...</td>
      <td>fear</td>
      <td>-1</td>
      <td>sit write start dig feeling think afraid accep...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ive been really angry with r and i feel like a...</td>
      <td>joy</td>
      <td>1</td>
      <td>ve angry r feel like idiot trust place</td>
    </tr>
    <tr>
      <th>4</th>
      <td>i feel suspicious if there is no one outside l...</td>
      <td>fear</td>
      <td>-1</td>
      <td>feel suspicious outside like rapture happen</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.preprocessed_comment, 
    df.label_Emotion, 
    test_size=0.2,
    random_state=2022,
    stratify=df.label_Emotion
)
```


```python
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
```

                  precision    recall  f1-score   support
    
              -1       0.97      0.91      0.94       388
               0       0.90      0.93      0.91       400
               1       0.90      0.93      0.92       400
    
        accuracy                           0.92      1188
       macro avg       0.92      0.92      0.92      1188
    weighted avg       0.92      0.92      0.92      1188
    
    


```python
#THE ABOVE MODEL SEEMS TO PERFORM REALLY WELL ON THE GIVEN TEXT AFTER PREPROCESSING WITH EXCELLENT ACCURACY, F1 SCORE AND RECALL.
#IT SIGNIFIES THE IMPORTANCE OF DATA PREPROCESSING ESPECIALLY IN TEXT ANALYTICS WHICH COULD SAVE TIME AND SPACE AND ULTIMATELY THE OPERATIONAL COST.
```


```python
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
```

                  precision    recall  f1-score   support
    
              -1       0.92      0.92      0.92       388
               0       0.92      0.89      0.90       400
               1       0.91      0.95      0.93       400
    
        accuracy                           0.92      1188
       macro avg       0.92      0.92      0.92      1188
    weighted avg       0.92      0.92      0.92      1188
    
    


```python

```
