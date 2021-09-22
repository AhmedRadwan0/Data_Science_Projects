import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import numpy as np
import pandas as pd 
import re
from sqlalchemy import create_engine
from sklearn.metrics import precision_score, accuracy_score, f1_score , classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = database_filepath[:-3].split('/')[-1]
    df = pd.read_sql_table(table_name, engine)
    X = df["message"]
    Y = df.drop(["id", "original", "message","genre"], axis =1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english") ]
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens]
    return clean_tokens


def build_model():
    
    pipeline = Pipeline([
    ('vec', CountVectorizer(tokenizer= tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [10, 20, 50] 
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_preds = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(category, classification_report(Y_test.iloc[:,i], Y_preds[:,i]))


def save_model(model, model_filepath):
    
    
    model_pkl = open(model_filepath, 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()



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
        evaluate_model(model, X_test, Y_test, category_names)

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