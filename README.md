# Data_Science_Projects
# Disaster Response Pipeline Project

# Project Overview
In the Project Workspace, we worked on a data set (provided by FigureEight) containing real messages (tweets) that were sent during disaster events.
The goal is creating a machine learning pipeline to categorize these events so that the messages can be sent to an appropriate disaster relief agency.
Your project includes a web app where an emergency worker can input a new message and get classification results in several categories. 
The web app will also display visualizations of the data.


# Instructions :
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# File structure:
```
- DisasterResponseProject
  - app
  | - template
  | |- master.html  # main page of web app
  | |- go.html  # classification result page of web app
  |- run.py  # Flask file that runs app

  - data
  |- disaster_categories.csv  # data to process (messages categories)
  |- disaster_messages.csv  # data to process (content of messages and their related categories)
  |- process_data.py        # python script runs ETL pipeline that cleans data and stores in database
  |- InsertDatabaseName.db   # database to save clean data to

  - models
  |- train_classifier.py   # python script runs ML pipeline that trains classifier, print the classification report for each category and saves the model
  |- classifier.pkl  # saved model 
```

