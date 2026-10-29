import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# LOAD PROCESSED DATASET
# ==========================================

df = pd.read_csv("dataset/processed_movies.csv")


# ==========================================
# CREATE TF-IDF MATRIX
# ==========================================

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(df["movie_features"])


# ==========================================
# CALCULATE COSINE SIMILARITY
# ==========================================

cosine_sim = cosine_similarity(tfidf_matrix)


# ==========================================
# PERSONALIZED RECOMMENDATION FUNCTION
# ==========================================

def personalized_recommendations(
    favorite_movie,
    preferred_genre="",
    preferred_director="",
    preferred_actor="",
    number_of_recommendations=5
):

    movie_matches = df[
        df["movie_name"].str.lower() == favorite_movie.lower()
    ]

    if movie_matches.empty:
        return []

    movie_index = movie_matches.index[0]

    similarity_scores = cosine_sim[movie_index]

    results = df.copy()

    results["similarity_score"] = similarity_scores

    results["personalized_score"] = results["similarity_score"]

    # Genre preference
    if preferred_genre:
        genre_match = results["genre"].str.lower().str.contains(
            preferred_genre.lower(),
            na=False
        )

        results.loc[genre_match, "personalized_score"] += 0.20

    # Director preference
    if preferred_director:
        director_match = results["director"].str.lower().str.contains(
            preferred_director.lower(),
            na=False
        )

        results.loc[director_match, "personalized_score"] += 0.20

    # Actor preference
    if preferred_actor:
        actor_match = results["cast"].str.lower().str.contains(
            preferred_actor.lower(),
            na=False
        )

        results.loc[actor_match, "personalized_score"] += 0.20

    # Remove favorite movie
    results = results.drop(index=movie_index)

    # Sort by personalized score
    results = results.sort_values(
        by="personalized_score",
        ascending=False
    )

    return results.head(number_of_recommendations)


# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================

st.title("🎬 AI-Based Movie Recommendation System")

st.write(
    "Get personalized movie recommendations based on "
    "your favorite movie and preferences."
)


# ==========================================
# USER INPUT
# ==========================================

st.subheader("Enter Your Preferences")

favorite_movie = st.selectbox(
    "Select your favorite movie:",
    sorted(df["movie_name"].dropna().unique())
)

preferred_genre = st.text_input(
    "Preferred Genre:",
    placeholder="Example: Action"
)

preferred_director = st.text_input(
    "Preferred Director:",
    placeholder="Example: Atlee"
)

preferred_actor = st.text_input(
    "Preferred Actor:",
    placeholder="Example: Shah Rukh Khan"
)


# ==========================================
# RECOMMENDATION BUTTON
# ==========================================

if st.button("🎯 Get Recommendations"):

    recommendations = personalized_recommendations(
        favorite_movie=favorite_movie,
        preferred_genre=preferred_genre,
        preferred_director=preferred_director,
        preferred_actor=preferred_actor,
        number_of_recommendations=5
    )

    if len(recommendations) == 0:

        st.error("Movie not found.")

    else:

        st.subheader("🍿 Recommended Movies")

        for rank, (_, movie) in enumerate(
            recommendations.iterrows(),
            start=1
        ):

            st.write(
                f"**{rank}. {movie['movie_name']}**"
            )

            st.write(
                f"Genre: {movie['genre']}"
            )

            st.write(
                f"Director: {movie['director']}"
            )

            st.write(
                f"Recommendation Score: "
                f"{movie['personalized_score']:.2f}"
            )

            st.divider()