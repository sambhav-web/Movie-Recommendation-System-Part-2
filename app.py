import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
with open('vectors.pkl','rb') as f:
  vectors=joblib.load(f)
df=pd.read_csv('movies.pkl')
def recommend(movie):
  index=df[df['title']==movie].index[0]
  vector=vectors[index].toarray()
  scores= cosine_similarity(vector,vectors).ravel()
  movie_list=sorted(enumerate(scores),reverse=True,key=lambda x: x[1])[1:6]
  indexes=[i for i,j in movie_list]
  movies=[df[df.index==i]['title'].values[0] for i in indexes]
  return movies

Movies=list(df['title'])
st.title('Movie Recommendation System')
st.info('This model predicts the similar movies as per choice. Sice it is frequency based model so can make wrong predictions.')
movie=st.selectbox('Choose Movie :',[None]+Movies)
if st.button('Suggest Options'):
  st.subheader(f'Similar movies to {movie} are :')
  recommendations=recommend("Jumanji")
  for i,j in enumerate(recommendations):
    st.subheader(f"{i} - {j}")
