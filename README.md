# AI-Based Movie Recommendation System

A Python project that recommends movies based on a user's favorite movie and preferences.

## Features
- Content-based movie recommendations
- TF-IDF text feature extraction
- Cosine similarity calculation
- Personalized recommendations based on genre, director, and actor
- Interactive Streamlit interface

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit

## How to Run
1. Install Python 3.
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure `dataset/processed_movies.csv` exists.
4. Run: `streamlit run app.py`

## Dataset Preparation
If the processed dataset is missing, ensure the original CSV is in the `dataset` folder and run `python preprocessing.py`.