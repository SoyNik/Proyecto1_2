import re
import nltk
import keras
import spacy
import inflect
import sent2vec # Para descargar esta libreria, es necesario descargarla desde GitHub https://github.com/epfml/sent2vec
import stopwords
import numpy as np
import unicodedata
import pandas as pd
import contractions
import seaborn as sns
nltk.download('wordnet')
nltk.download('omw-1.4')
import pandas_profiling as pp
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from sklearn.utils import resample 
from nltk import word_tokenize, sent_tokenize
from tensorflow.keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.layers import LSTM, Dense, Embedding, TextVectorization, Input, Dropout



    
# Ahora, creamos todas las funciones de preprocesamiento de textos.
    
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words
    

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_numbers(words):
    p = inflect.engine()
    new_words = []
    for word in words:
        new_word = re.sub('\d+.*', '', word)
        if not word.isnumeric() and new_word != '':
            new_words.append(word)
    return new_words

def remove_dates(words):
    """Replace all dates in our data"""
    new_words = []
    for word in words:
        new_word = re.sub(r'\d+/\d+/\d+', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


our_stopwords = ["paty","patients","p","study","result", "human", "humans", "monkey", "monkeys", 
                 "diseases", "studied","first", "rat", "patient", "case", "p less", "treatment", 
                 "group", "associated", "result", "may", "effect", "compared", "use", "cases", "year", 
                 "years", "age", "study", "disease", "found", "normal", "month", "although", "per cent",
                 "one", "two", "three", "four", "n", "children", "women"]

def remove_stopwords(words):
    new_words = []
    for word in words:
        if word not in stopwords.words('english') and word not in our_stopwords:
            new_words.append(word)
    return new_words

def noise_elimination(words):
    words = to_lowercase(words)
    words = remove_non_ascii(words)
    words = remove_numbers(words)
    words = remove_dates(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words

# Finalmente, creamos una clase para esta parte del preprocesamiento
def preprocessing(X):
    new_X_train= X.apply(contractions.fix) #Aplica la corrección de las contracciones
    new_X_train = new_X_train.apply(word_tokenize)
    new_X_train = new_X_train.apply(noise_elimination) #Aplica la eliminación del ruido
    X_train = new_X_train.apply(lambda x: ' '.join(map(str, x)))
    return new_X_train, X_train

class Preprocessing():
    def __init__(self):
        pass
    def transform(self,X,y=None):
        X["medical_abstracts"] = X["medical_abstracts"].apply(contractions.fix)
        X["medical_abstracts"] = X["medical_abstracts"].apply(word_tokenize)
        X["medical_abstracts"] = X["medical_abstracts"].apply(noise_elimination)
        return X
    def fit(self, X, y=None):
        return self

# Stemming y lematizacion
#Funciones de "Stemming" y "Lemmatization"
def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = SnowballStemmer('english')
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems + lemmas

class StemAndLemmatize():
    def __init__(self):
        pass
    def transform(self,X,y=None):
        X["medical_abstracts"] = X["medical_abstracts"].apply(stem_and_lemmatize)
        #X["non_tokenized_abstracts"] = X["medical_abstracts"].apply(lambda x: ' '.join(map(str, x)))
        return X
    def fit(self, X, y=None):
        return self

    
# ----- PARA ENTRENAMIENTO -----

class VectorizeLSTM():
    def __init__(self):
        pass
    def transform(self,X,y=None):
        X["medical_abstracts"] = X["medical_abstracts"].apply(lambda x: ' '.join(map(str, x)))
        model_path = '../notebooks/BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
        model = sent2vec.Sent2vecModel()
        try:
            model.load_model(model_path)
            print('Model successfuly loaded')
        except Exception as e:
            print("HOLA", e)
        new_X = model.embed_sentences(X["medical_abstracts"])
        new_X = new_X.reshape(-1, 1, new_X.shape[1])
        return new_X
    def fit(self, X, y=None):
        return self
    
class LSTMBuilder():
    def __call__(self):
        output=5
        model = Sequential(name="LSTM")
        model.add(LSTM(units=64, return_sequences=True,
                    input_shape=(1, 700)))
        model.add(Dropout(0.1))
        model.add(LSTM(units=64, return_sequences=False))
        model.add(Dropout(0.1))
        model.add(Dense(output, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision(name='precision')])
        return model

    
    
    

    
        
        
        