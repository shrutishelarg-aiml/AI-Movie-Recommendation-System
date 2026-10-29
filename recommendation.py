import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load processed dataset
df = pd.read_csv("dataset/processed_movies.csv")

print("Processed dataset loaded!")
print("Number of movies:", len(df))

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words="english")

# Convert movie features into numerical vectors
tfidf_matrix = tfidf.fit_transform(df["movie_features"])

print("\n===== TF-IDF INFORMATION =====")
print("Number of movies:", tfidf_matrix.shape[0])
print("Number of features:", tfidf_matrix.shape[1])
print("TF-IDF matrix shape:", tfidf_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity between all movies
cosine_sim = cosine_similarity(tfidf_matrix)

print("\n===== COSINE SIMILARITY =====")
print("Similarity matrix shape:", cosine_sim.shape)

# Display similarity between the first movie and the next 5 movies
print("\nSimilarity scores for the first movie:")
print(cosine_sim[0][:5])


# ==========================================
# STEP 6: MOVIE RECOMMENDATION FUNCTION
# ==========================================

def recommend_movies(movie_name, number_of_recommendations=5):

    # Find the movie
    movie_matches = df[
        df["movie_name"].str.lower() == movie_name.lower()
    ]

    # Check if movie exists
    if movie_matches.empty:
        print("\nMovie not found in the dataset.")
        return

    # Get the index of the selected movie
    movie_index = movie_matches.index[0]

    # Get similarity scores for the selected movie
    similarity_scores = list(enumerate(cosine_sim[movie_index]))

    # Sort movies by similarity score
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Select top recommendations
    top_movies = similarity_scores[1:number_of_recommendations + 1]

    print("\n===== RECOMMENDED MOVIES =====")
    print("Based on:", df.loc[movie_index, "movie_name"])

    for rank, (index, score) in enumerate(top_movies, start=1):

        movie_title = df.loc[index, "movie_name"]

        print(
            f"{rank}. {movie_title} "
            f"(Similarity: {score:.2f})"
        )

# ==========================================
# STEP 8: PERSONALIZED RECOMMENDATION
# ==========================================

def personalized_recommendations(
    favorite_movie,
    preferred_genre="",
    preferred_director="",
    preferred_actor="",
    number_of_recommendations=5
):

    # Find the favorite movie
    movie_matches = df[
        df["movie_name"].str.lower() == favorite_movie.lower()
    ]

    if movie_matches.empty:
        print("\nFavorite movie not found in the dataset.")
        return

    # Get movie index
    movie_index = movie_matches.index[0]

    # Get similarity scores
    similarity_scores = cosine_sim[movie_index]

    # Create a copy of the dataset for scoring
    results = df.copy()

    # Store similarity score
    results["similarity_score"] = similarity_scores

    # Start personalized score with similarity
    results["personalized_score"] = results["similarity_score"]

    # Add genre preference
    if preferred_genre:
        genre_match = results["genre"].str.lower().str.contains(
            preferred_genre.lower(),
            na=False
        )

        results.loc[genre_match, "personalized_score"] += 0.20

    # Add director preference
    if preferred_director:
        director_match = results["director"].str.lower().str.contains(
            preferred_director.lower(),
            na=False
        )

        results.loc[director_match, "personalized_score"] += 0.20

    # Add actor preference
    if preferred_actor:
        actor_match = results["cast"].str.lower().str.contains(
            preferred_actor.lower(),
            na=False
        )

        results.loc[actor_match, "personalized_score"] += 0.20

    # Remove the favorite movie itself
    results = results.drop(index=movie_index)

    # Sort according to personalized score
    results = results.sort_values(
        by="personalized_score",
        ascending=False
    )

    # Display recommendations
    print("\n==========================================")
    print("   PERSONALIZED RECOMMENDATIONS")
    print("==========================================")

    print("Favorite Movie:", favorite_movie)

    if preferred_genre:
        print("Preferred Genre:", preferred_genre)

    if preferred_director:
        print("Preferred Director:", preferred_director)

    if preferred_actor:
        print("Preferred Actor:", preferred_actor)

    print("\nTop Recommendations:")

    for rank, (_, movie) in enumerate(
        results.head(number_of_recommendations).iterrows(),
        start=1
    ):

        print(
            f"{rank}. {movie['movie_name']} "
            f"(Score: {movie['personalized_score']:.2f})"
        )

# ==========================================
# STEP 8.3: TEST PERSONALIZED RECOMMENDATION
# ==========================================

personalized_recommendations(
    favorite_movie="Jawan",
    preferred_genre="Action",
    preferred_director="Atlee",
    preferred_actor="Shah Rukh Khan",
    number_of_recommendations=5
)