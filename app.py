import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
import requests
import os

key = os.environ.get("TMDB_API_KEY")

with open('vectors.pkl','rb') as f:
  vectors=joblib.load(f)
df=pd.read_csv('movies.csv')

def api_call(movie_id):
  url="https://api.themoviedb.org/3/movie/"
  image_url="https://image.tmdb.org/t/p/w500/"
  params={ "api_key":key}
  L=[]
  for i in movie_id:
    index=df[df.index==i]['id']
    response=requests.get(url+f"{index}",params=params)
    try:
      image_path=response.json()['poster_path']
    except Exception as e:
      pass
    L.append(f'image{i+1}.jpg')
    response2=requests.get(image_url+image_path)
    with open(f'image{i+1}.jpg','wb') as f:
      f.write(response2.content)
  return L

def recommend(movie):
  index=df[df['title']==movie].index[0]
  vector=vectors[index].toarray()
  scores= cosine_similarity(vector,vectors).ravel()
  movie_list=sorted(enumerate(scores),reverse=True,key=lambda x: x[1])[1:6]
  indexes=[i for i,j in movie_list]
  images=api_call(indexes)
  movies=[df[df.index==i]['title'].values[0] for i in indexes]
  return movies,images

Movies=list(df['title'])
st.title('Movie Recommendation System')
st.info('💡 **Note:** This model generates recommendations based on keyword frequency. Results may vary based on similarity scores.')
movie=st.sidebar.selectbox('Choose Movie :',[None]+Movies)
if st.button('Suggest Movies'):
  if movie is None:
    st.warning('⚠️ Please select a movie')
  else:
    st.subheader(f'Similar movies to {movie} are :')
    recommendations,images=recommend(movie)
    for i,j in enumerate(images):
      with open(j,'rb') as f:
        image=f.read()
      st.image(image, caption=recommendations[i], width=400)
