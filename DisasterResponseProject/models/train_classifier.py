import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import pandas as pd 
import re
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle


def load_data(database_filepath):
    """
    Load data from database
    
    Input: database_filepath - String
           The file path of the database
                    
    Output: X - Pandas DataFrame
            Feature variables to be used in machine learning
            y - Pandas DataFrame
            Response variables (multioutput labels to be predicted)
            
            category_names - List
            List of the y response variable column names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = database_filepath[:-3].split('/')[-1]
    df = pd.read_sql_table(table_name, engine)
    X = df["message"]
    Y = df.drop(["id", "original", "message","genre"], axis =1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
       Tokenizer that cleans the text and break it into matrix for machine learning
    """
    
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english") ]
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens]
    return clean_tokens


def build_model():
    """
    Build the pipeline model to be used as the model
    
    Output: cv - model
            The model structure to be used for fitting and predicting
    """
    
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
     """
    Evalueate the performance of the model.
    Prints out the classification report of each response variable (category)
    """
    Y_preds = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(category, classification_report(Y_test.iloc[:,i], Y_preds[:,i]))


def save_model(model, model_filepath):
     """
      Saves the model to the specified model path
    """
    model_pkl = open(model_filepath, 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()



def main():
     """
    Create machine learning models and save output to pickle file
    """
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