import sys
from sqlalchemy import create_engine
from pandas import Series, DataFrame 
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re 
import nltk
import pickle

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import classification_report


def load_data(database_filepath):
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM Messages", engine)

    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    # Create list containing all category names 
    category_names = list(Y.columns.values)
    
    return X, Y, category_names


def tokenize(text):
    
    
    # convert text to lowercase and remove punctuation
    test = re.sub(r"[^a-zA-Z0-9]", "", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # remove stop words 
    stop_words = stopwords.words("english")
    tokens_stop = [w for w in tokens if w not in stop_words]
    
    # lemmatize 
    lemmatizer = WordNetLemmatizer()
    tokens_clean = []
    for tok in tokens_stop:
        tok_clean = lemmatizer.lemmatize(tok).strip()
        tokens_clean.append(tok_clean)
    
    return tokens_clean


def build_model():
    
    
    pipeline =  Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'tfidf__use_idf':[True],
    'clf__estimator__n_estimators':[20]
    }
    
    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def model_results(model, X_test, Y_test, category_names):
    #Testing the model
    # Printing the classification report for each label
    
    Y_pred = model.predict(X_test)
    i = 0
    for col in Y_test:
        print('Feature {}: {}'.format(i+1, col))
        print(classification_report(Y_test[col], Y_pred[:, i]))
        i = i + 1
    accuracy = (Y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))
    


def save_model(model, model_filepath):
        # create a pickle file for the model 

    with open (model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        model_results(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()