import os
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template

# Load your data
df = pd.read_csv('movies_new.csv')

# Fill missing values in selected columns
selected = ['genres', 'keywords', 'popularity', 'tagline', 'cast', 'director']
for feature in selected:
    df[feature] = df[feature].fillna('')

# Combine selected features into a single string
combined = (df['genres'].astype(str) + ' ' +
            df['keywords'].astype(str) + ' ' +
            df['popularity'].astype(str) + ' ' +
            df['tagline'].astype(str) + ' ' +
            df['cast'].astype(str) + ' ' +
            df['director'].astype(str))

# Create a TF-IDF Vectorizer and compute the feature vectors
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined)

# Compute the cosine similarity matrix
similarity = cosine_similarity(feature_vector)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    movie_name = request.form.get('movie_name', '')
    if not movie_name:
        return render_template('index.html', error="No movie name provided")
    
    list_title = df['title'].tolist()
    close_match = difflib.get_close_matches(movie_name, list_title)
    
    if not close_match:
        return render_template('index.html', error="Movie not found", movie_name=movie_name)
    
    closer = close_match[0]
    index_movie = df[df.title == closer]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_movie]))
    sorted_similar = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, movie in enumerate(sorted_similar[:30], 1):
        index = movie[0]
        title_fromindex = df[df.index == index]['title'].values[0]
        recommendations.append((i, title_fromindex))
    
    return render_template('index.html', recommendations=recommendations, movie_name=movie_name)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=False for production

