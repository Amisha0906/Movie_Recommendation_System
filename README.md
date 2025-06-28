## 🎬 Movie Recommendation System

This project demonstrates two major approaches to building a movie recommendation engine:

Content-Based Filtering - a recommendation technique that suggests items to users based on the features (or content) of those items.
Collaborative Filtering - a recommendation technique to predict user preferences for items based on the preferences of similar users.

## 📊 Datasets Used

[MovieLens Dataset]([url](https://grouplens.org/datasets/movielens/))

## 📈 Approach

### 🔹 Content-Based Filtering

1. Preprocessing the Genres

2. Creating TF-IDF Matrix - Converting genres into numerical features using the TF-IDF Vectorizer, which transforms genre text data into numerical feature vectors

3. Training KNN model - Using the k-Nearest Neighbors algorithm with cosine similarity to identify similar movies.

4. Recommended similar movies based on the selected title

### 🔹 Collaborative Filtering

1. Creating a User-Movie Matrix - Since we're comparing users, this matrix helps us represent user behavior in a structured format, making it easy to compare users based on the movies they’ve rated
2. Calculating Similarity Between Users - Cosine Similarity and Pearson Correlation
3. Making Predictions and Recommendations -
   
      Finding top-k similar users,

      Predicting movie ratings using weighted averages,
  
      Recommending the top N movies that the target user hasn’t seen

## 🧾Medium Articles

[Content based filtering]([url](https://medium.com/@amishasaha/building-a-movie-recommendation-system-part-i-5fc694406a85))

[Collaborative filtering]([url](https://medium.com/@amishasaha/building-a-movie-recommendation-system-part-ii-514e1ec75db7))
