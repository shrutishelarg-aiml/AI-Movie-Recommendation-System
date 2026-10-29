import pandas as pd

# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/IMDB-Movie-Dataset(2023-1951).csv")

print("Original dataset shape:", df.shape)


# ==========================================
# STEP 2: REMOVE UNNECESSARY COLUMN
# ==========================================

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("\nAfter removing unnecessary column:")
print(df.shape)


# ==========================================
# STEP 3: HANDLE MISSING VALUES
# ==========================================

df["year"] = df["year"].fillna("Unknown")


# ==========================================
# STEP 4: CLEAN TEXT COLUMNS
# ==========================================

text_columns = [
    "genre",
    "overview",
    "director",
    "cast"
]

for column in text_columns:
    df[column] = df[column].fillna("")
    df[column] = df[column].astype(str).str.strip()


# ==========================================
# STEP 5: CREATE MOVIE FEATURES
# ==========================================

df["movie_features"] = (
    df["genre"] + " " +
    df["overview"] + " " +
    df["director"] + " " +
    df["cast"]
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\nFinal dataset shape:", df.shape)

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nColumns:")
print(df.columns.tolist())

print("\nSample movie features:")
print(
    df[["movie_name", "genre", "director", "movie_features"]].head()
)


# ==========================================
# SAVE PROCESSED DATASET
# ==========================================

df.to_csv(
    "dataset/processed_movies.csv",
    index=False
)

print("\nProcessed dataset saved successfully!")