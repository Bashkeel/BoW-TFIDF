
# TF-IDF and N-Grams
The goal of this project was to predict the sentiment of an IMDB movie review using a binary classification system. The dataset was part of the [Bag of Words Meets Bag of Popcorn Competition](https://www.kaggle.com/c/word2vec-nlp-tutorial).

<b>Model Accuracy: 0.89532</b>

## Bag of Words & TF-IDF

A Bag of Words (BoW) model is a simple algorithm used in Natural Language Processing. It simply counts the number of times a word appears in a document.

TF-IDF (or Term Frequency-Inverse Document Frequency) on the other hand reflects how important a word is to a document, or corpus. With TF-IDF, words are given weight, measured by <i>relevance</i>, rather than <i>frequency</i>.


It is the product of two statistics:
1. <b>Term Frequency</b> (TF): The number of times a word appears in a given document.
2. <b>Inverse Document Frequency</b> (IDF): The more documents a word appears in, the less valuable that word is as a signal. Very common words, such as “a” or “the”, thereby receive heavily discounted tf-idf scores, in contrast to words that are very specific to the document in question.

<img src="https://skymind.ai/images/wiki/tfidf.png">

In the project, I used two separate TF-IDF vectorizers and merged them into a single bag of words.
* The first vectorizer (<i>word_vectorizer</i>) analyzed complete words.
* The second vectorizer (<i>char_vectorizer</i>) analyzed the frequency of character n-grams. An <i>n-gram</i> is a continous sequence of <i>n</i> items from a document. Using Trigrams (<i>N-gram size = 3</i>) yielded a high predictive score.

Lastly, we used a Logistic Regression to predict the sentiment attached to each review. The hyperparameters of the model were tuned using a validation dataset prior to training the model.

#### Interestingly, our model performed <b>worse</b> if we cleaned the text data in the usual methods. This includes removing html, removing unwanted punctuation, removing stopwords, stemming, or tokenizing.

## Loading Required Libraries and Reading the Data into Python

First, we need to load the required libraries and read the data into Python.


```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.sparse import hstack
from time import time
```


```python
train = pd.read_csv( "labeledTrainData.tsv", header=0, delimiter="\t")
test = pd.read_csv("testData.tsv", header=0, delimiter="\t")

train_text = train['review']
test_text = test['review']
y = train['sentiment']

all_text = pd.concat([train_text, test_text])
```

## TF-IDF Vectorizers
First, we convert the reviews into a Bag of Words using the TF-IDF vectorizer for words and for character trigrams.


```python
word_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', sublinear_tf=True, strip_accents='unicode',
                                  stop_words='english', ngram_range=(1, 1), max_features=10000)
word_vectorizer.fit(train_text)

train_word_features = word_vectorizer.transform(train_text)
```


```python
char_vectorizer = TfidfVectorizer(analyzer='char', sublinear_tf=True, strip_accents='unicode',
                                  stop_words='english', ngram_range=(1, 3), max_features=50000)
char_vectorizer.fit(train_text)

train_char_features = char_vectorizer.transform(train_text)
```


```python
train_features = hstack([train_word_features, train_char_features])
```

## Hyperparameter Tuning of Logistic Regression
Since there are multiple hyperparameters to tune in the XGBoost model, we will use the [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) function of Sklearn to determine the optimal hyperparameter values. Next, I used the [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function to generate a validation set and find the best parameters.


```python
X_train, X_test, y_train, y_test = train_test_split(train_features, y,test_size=0.3 ,random_state=1234)

lr_model = LogisticRegression(random_state=1234)
param_dict = {'C': [0.001, 0.01, 0.1, 1, 10],
             'solver': ['sag', 'lbfgs', 'saga']}

start = time()
grid_search = GridSearchCV(lr_model, param_dict)
grid_search.fit(X_train, y_train)
print("GridSearch took %.2f seconds to complete." % (time()-start))
display(grid_search.best_params_)
print("Cross-Validated Score of the Best Estimator: %.3f" % grid_search.best_score_)
```

    GridSearch took 350.08 seconds to complete.



    {'C': 1, 'solver': 'saga'}


    Cross-Validated Score of the Best Estimator: 0.888


Let's see how well our model does on the validation dataset and where any misclassifications occur.

We have several metrics available for classification accuracy, including a confusion matrix and a classification report.


```python
lr=LogisticRegression(C=1, solver ='saga')
lr.fit(X_train, y_train)
lr_preds=lr.predict(X_test)

print(confusion_matrix(y_test, lr_preds))
print(classification_report(y_test, lr_preds))
print("Accuracy Score: %.3f" % accuracy_score(y_test, lr_preds))
```

    [[3366  399]
     [ 366 3369]]
                 precision    recall  f1-score   support

              0       0.90      0.89      0.90      3765
              1       0.89      0.90      0.90      3735

    avg / total       0.90      0.90      0.90      7500

    Accuracy Score: 0.898


The number of false positives (<i>FP = 366</i>) is similar to the number of false negatives (<i>FN = 399</i>), suggesting that our model is not biased towards either specificity nor sensitivity.

## Modelling Sentiment from Reviews
We will redo the steps taken above, this time we both the train and test dataset.

1. Create a TF-IDF BoW for both words and trigrams.
2. Train the Logistic Regression model using the tuned hyperparameters.
3. Format predictions for submission to Kaggle Competition.


```python
word_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', sublinear_tf=True, strip_accents='unicode',
                                  stop_words='english', ngram_range=(1, 1), max_features=10000)
word_vectorizer.fit(all_text)

train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)
```


```python
char_vectorizer = TfidfVectorizer(analyzer='char', sublinear_tf=True, strip_accents='unicode',
                                  stop_words='english', ngram_range=(1, 3), max_features=50000)
char_vectorizer.fit(all_text)

train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)
```


```python
train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])
```


```python
lr=LogisticRegression(C=1,solver='saga')
lr.fit(train_features,y)
final_preds=lr.predict(test_features)
```

The predictions are then formatted in an appropriate layout for submission to Kaggle.


```python
test['sentiment'] = final_preds
test = test[['id', 'sentiment']]
test.to_csv('Submission.csv',index=False)
```

### Logistic Regression Sentiment  Accuracy = 0.87592
