from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
import pickle
import pandas as pd
import re
import numpy as np

app = Flask(__name__)

movies = pd.read_csv("ml-latest/movies.csv")
movies['genres'] = movies['genres'].fillna('').str.replace('|', ' ')
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
model = pickle.load(open("knn_movie.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


def recommend_movies(title, model, data, n_recommendations=10):
    title = movies[movies['title'].str.contains(title, case=False, na=False)]['title'].values[0]
    if title not in indices:
        return ["Movie not found"]
    idx = indices[title]
    distances, indices_list = model.kneighbors(data[idx], n_neighbors=n_recommendations+1)
    movie_indices = indices_list[0][1:]  # Exclude the first one (itself)
    # return movies['title'].iloc[movie_indices]
    return list(movies['title'].iloc[movie_indices])


@app.route("/predict", methods = ["POST"])
def predict():
    text = request.form["text"]
    results = recommend_movies(title=str(text), model=model, data=tfidf_matrix)
    return render_template("index.html", pred="Recommendations: " + ", ".join(results))

   #  return render_template("index.html", pred="Similar movies: {}".format(my_pred))


if __name__ =="__main__":
    # app.run(debug=True)
    app.run(debug=True, port=5001)