# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import scipy.sparse
import csv

import pickle
import streamlit as st

import spacy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim import models
from gensim.models import Phrases
from gensim.models.phrases import Phraser


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

######################################### LOAD DỮ LIỆU VÀ MÔ HÌNH ###########################################

################################ Load danh sách stop word ###################################################
stop_words = list(pd.read_csv('stop_words.csv'))

################################ Load các mô hình tiền xử lý ################################################
def analyzer(x):
    return x

# Mô hình tiền xử lý ngôn ngữ của Spacy (dùng cho kỹ thuật Lemmatization)
with open('spacy_nlp.pkl', 'rb') as f:
    spacy_nlp = pickle.load(f)

# Mô hình tạo các từ ghép 2 chữ (Bigrams)
with open('bigrams_phraser.pkl', 'rb') as f:
    bigrams_phraser = pickle.load(f)
    
# Mô hình tạo các từ ghép 3 chữ (Trigrams)
with open('trigrams_phraser.pkl', 'rb') as f:
    trigrams_phraser = pickle.load(f)

# Mô hình vectơ hóa bằng kỹ thuật Bag of Word
with open('bow_vectorizer.pkl', 'rb') as f:
    bow_vectorizer = pickle.load(f)
    
# Mô hình vectơ hóa bằng kỹ thuật TF-IDF
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

################################ Load các mô hình học máy ################################################
# Tên mô hình
model_names = ['SVM', 
               'DecisionTree', 
               'LogisticRegression', 
               'NaiveBayes']

# Load các mô hình học máy cho bộ dữ liệu Bag of Word
bow_models = {}
for name in model_names:
    with open('bow_'+name+'.pkl', 'rb') as file:
        bow_models[name] = pickle.load(file)

# Load các mô hình học máy cho bộ dữ liệu TF-IDF
tfidf_models = {}
for name in model_names:
    with open('tfidf_'+name+'.pkl', 'rb') as file:
        tfidf_models[name] = pickle.load(file)


################ TẠO CÁC HÀM #####################
# Hàm tiền xử lý văn bản
def preprocess_corpus(corpus):
    # Thực hiện kỹ thuật Lemmatization
    doc = spacy_nlp(corpus)
    lemmatized_tokens = [token.lemma_ for token in doc]

    # Loại bỏ stop_words
    nostop_tokens = []
    for word in lemmatized_tokens:
        w = word.lower()
        if w.lower() not in stop_words and w.isalpha():
            nostop_tokens.append(w)

    # Phân tích Bigrams và Trigrams
    bigrams = list(bigrams_phraser[nostop_tokens])
    trigrams = list(trigrams_phraser[bigrams])
    return trigrams

# Hàm vector hóa văn bản
def vectorize(trigrams,text_model):
    if text_model == 'Bag of Words':
        bow_vector = bow_vectorizer.transform([trigrams])
        return bow_vector
    if text_model == 'TF-IDF':
        tfidf_vector = tfidf_vectorizer.transform([trigrams])
        return tfidf_vector

# Hàm dự đoán
def label_prediction(vector,text_model,class_model):
    
    if text_model == 'Bag of Words':
        label = bow_models[class_model].predict(vector)
        return label[0]
    if text_model == 'TF-IDF':
        label = tfidf_models[class_model].predict(vector)
        return label[0] 
    
def main():
    
    st.title('Review Prediction')
    
    corpus = st.text_area("Enter Review")
    text_model = st.radio(
        'Choose the model for text preprocessing',
        ['Bag of Words','TF-IDF'])
    class_model = st.selectbox(
        'Choose the model for classification',
        ('SVM','DecisionTree','LogisticRegression','NaiveBayes'),
        index=None,
        placeholder='Select classification model...')
    label = ''
    if st.button('Predict!'):
        trigrams = preprocess_corpus(corpus)
        vector = vectorize(trigrams,text_model)
        label = loan_prediction(vector,text_model,class_model) 

    st.success(label)    
    


if __name__ == '__main__':
    main()    
