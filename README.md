# Text-Classification

## **Libraries and Packages**

**Install NLTK platform and sklearn library**

Import tokenize module from nltk for splliting the setences into words. Commonly used words such as
'a', 'an', 'the' etc are known as Stop words are removed using the corpus module of nltk. Stemming process
uses the stem module from nltk. Expressing the multiple words that occur commonly can be done using
collocation module of nltk.
Import the modules like feature extraction for converting corpus into bag of words and tfIdf representation.
Text-classification using NaiveBayes and Support Vector Machine from sklearn modules. Confusion
matrices are reported for test sets using the confusion matrix module of sklearn.
Before using nltk platform, download the required libraries using below commands in jupyter

    import nltk
    nltk.download()

**Import Other Libraries**

    from sklearn.datasets import fetch_20newsgroups
    from pprint import pprint
    from nltk.tokenize import word_tokenize
    import nltk
    from nltk.probability import FreqDist
    from nltk.collocations import *
    from nltk.corpus import stopwords
    import re
    import pandas as pd
    from nltk.stem import PorterStemmer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import SGDClassifier
    from sklearn import svm
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plot

## Data Cleaning

Data cleaning is required before creating the bigrams otherwise bigrams result may consist special characters
and unnecessary words which reduce the significance of creating the bigram. Following Operations are performed to clean the data.

1. Remove the stop words ('are','is','everything' etc)
2. Remove the word with a single character (e.g. 'c','a' etc)
3. Remove special character and numbers ('@','.',')','(' etc)
4. Remove all the null values

## Collocation

It is considered as a phrase which consists of multiple words that are co-occur in the text. For ex. In the set of
education related document word 'Machine learning' co-occur together rather than individual word 'Machine'
and 'Learning'.

**Bigrams**
Bigram is concatenation of two words which is considered as an individual words which helps to improve insight
analysis of the text document while solving NLP related problems.
We have used below four techniques to filter out meaningful collocations from the document.

1. Frequency Counting
2. Pointwise Mutual Information (PMI)
3. T-test
4. Chi-square

 

## Text classification.

 1. convert the corpus into a bag-of-words tf-idf weighted vector representation
 2. Split the data randomly into training and testing set (70-30 %).
 3. Trained SVM and reported confusion matrix.
 4. Trained Multinomial NB and reported confusion matrix
 5. Compared Accurecy after changing Kernel for SVM
