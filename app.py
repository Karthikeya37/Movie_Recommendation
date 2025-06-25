import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

@st.cache_data
def load_data():
    df1 = pd.read_csv('tmdb_5000_credits.csv')
    df2 = pd.read_csv('tmdb_5000_movies.csv')
    df1.columns = ['id','tittle','cast','crew']
    df2 = df2.merge(df1, on='id')

    df2['overview'] = df2['overview'].fillna('')
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df2['overview'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()
    
    return df2, cosine_sim, indices

def get_recommendations(title, df, cosine_sim, indices):
    if title not in indices:
        return []
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎥 Movie Recommendation System")

df, cosine_sim, indices = load_data()

movie_list = df['title'].sort_values().unique()
selected_movie = st.selectbox("Select a movie to get similar recommendations:", movie_list)

if st.button("Get Recommendations"):
    recommendations = get_recommendations(selected_movie, df, cosine_sim, indices)
    if recommendations:
        st.subheader("🔍 Recommended Movies:")
        for i, movie in enumerate(recommendations, start=1):
            st.write(f"{i}. {movie}")
    else:
        st.warning("No recommendations found. Try another movie.")

if st.checkbox("Show Top 5 Most Popular Movies"):
    top_popular = df.sort_values("popularity", ascending=False)[['title', 'popularity']].head(5)
    st.dataframe(top_popular.reset_index(drop=True))
