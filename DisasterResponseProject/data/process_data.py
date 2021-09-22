import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loading the data from CSV Files
    Input: messages_filepath - String
            The path to the messages data csv
           categories_filepath - String
            The path to the categories data csv
    Output: df = DataFrame
            A merged DataFrame of messages and categories data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on = ["id"])
    return df


def clean_data(df):
    """
    Extracts categories from categories data, converts values to 0's and 1's and removes duplicates
    Input: df - DataFrame
            Dataframe output from load_data function
    Output: df - DataFrame
            Cleansed dataframe of the input data
    """
    categories = df["categories"].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = list(row.apply(lambda cat: cat[:-2]))
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string (0 or 1)
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df = df.drop(["categories"], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(["message"], inplace=True)
    
    return df
    

def save_data(df, database_filename):
    """
    Save data to database
    Input: df - DataFrame
            DataFrame from clean_data dataframe
           database_filename - String
           Database file location of where data is to be stored
    """       

    engine = create_engine('sqlite:///' + database_filename)
    
    # Extracting the table name
    table_name = database_filename[:-3].split('/')[-1]
    
    df.to_sql(table_name, engine, index=False, if_exists ='replace')
    
    testframe = pd.read_sql("SELECT * FROM "+ table_name, engine)
    print(testframe.head())


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()