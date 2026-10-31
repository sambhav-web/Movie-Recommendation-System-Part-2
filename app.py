import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
import requests
from google.colab import userdata

key = userdata.get('API')

with open('vectors.pkl','rb') as f:
  vectors=joblib.load(f)
df=pd.read_csv('movies.csv')

def api_call(movie_id):
  url="https://api.themoviedb.org/3/movie/"
  image_url="https://image.tmdb.org/t/p/w500/"
  params={ "api_key":key}
  for i in movie_id:
    response=requests.get(url+f"{i}",params=params)
    image_path=response.json()['belongs_to_collection']['poster_path']
    response2=requests.get(image_url+image_path)
    with open(f'image{i+1}.jpg','wb') as f:
      f.write(response2.content)

def recommend(movie):
  index=df[df['title']==movie].index[0]
  vector=vectors[index].toarray()
  scores= cosine_similarity(vector,vectors).ravel()
  movie_list=sorted(enumerate(scores),reverse=True,key=lambda x: x[1])[1:6]
  indexes=[i for i,j in movie_list]
  api_call(indexes)
  movies=[df[df.index==i]['title'].values[0] for i in indexes]
  return movies

Movies=list(df['title'])
st.title('Movie Recommendation System')
st.info('💡 **Note:** This model generates recommendations based on keyword frequency. Results may vary based on similarity scores.')
movie=st.sidebar.selectbox('Choose Movie :',[None]+Movies)
if st.button('Suggest Movies'):
  if movie is None:
    st.warning('⚠️ Please select a movie')
  else:
    st.subheader(f'Similar movies to {movie} are :')
    recommendations=recommend(movie)
    for i,j in enumerate(recommendations):
      with open(f'image{i+1}.jpg','rb') as f:
        image=f.read()
      st.image(image, caption=j, width=400)
