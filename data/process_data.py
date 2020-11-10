import sys
import math
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    #Load messages dataset 
    messages = pd.read_csv(messages_filepath)
    
    #Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how='left', on=['id'])
    
    return df

def clean_data(df):
    
    # create a dataframe of the 36 individual category columns
    categories_col = df['categories'].str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories_col.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    col_names = row.apply(lambda x: x[:-2]).tolist()
    
    categories_col.columns = col_names
    
    # Convert category values to numeric values
    for column in categories_col:
        # set each value to be the last character of the string
        categories_col[column] = categories_col[column].transform(lambda x: x[-1])
        
        # convert column from string to numeric 
        categories_col[column] = pd.to_numeric(categories_col[column])
          
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories_col], axis=1)
    
    # drop the values 2 of 'related' feature
    df = df[df['related'] != 2]
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')
    


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