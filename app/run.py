import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_related = df[df['related']==1].groupby('genre').count()['message']
    genre_not_related= df[df['related']==0].groupby('genre').count()['message']
    genre_names = list(genre_related.index)
    
    # Calculate proportion of each category with label = 1
    cat_props = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()/len(df)
    cat_props = cat_props.sort_values(ascending=False)
    cat_names = list(cat_props.index)
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_related,
                    name = 'Related'
                ),
                
                Bar(
                    x=genre_names,
                    y=genre_not_related,
                    name = 'Not Related'
                )
            ],

            'layout': {
                'title': 'Distribution of Messages by Genre and Related Status',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode': 'group'
            }
        },
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_props
                )
            ],

            'layout': {
                'title': 'Proportion of Messages by Category',
                'yaxis': {
                    'title': "Proportion"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
