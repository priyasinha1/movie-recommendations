
# In[1]:


import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process


# In[2]:


movies = 'movies.csv'
ratings = 'ratings.csv'


# In[3]:


df_movies = pd.read_csv(movies, usecols=['movieId','title'], dtype= {'movieId':'int32','title':'str'})


# In[4]:


df_movies.head()


# In[5]:


df_movies.describe()


# In[6]:


df_ratings = pd.read_csv(ratings, usecols=['userId','movieId','rating'], dtype = {'userId':'int32','movieId':'int32','rating':'float32'})
df_ratings.head()


# In[7]:


movies_users= df_ratings.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)
movies_users


# In[8]:


mat_movies_user = csr_matrix(movies_users.values)


# In[9]:


model_knn = NearestNeighbors(metric='cosine',algorithm = 'brute',n_neighbors = 20)


# In[10]:


model_knn.fit(mat_movies_user)


# In[11]:


pip install fuzzywuzzy


# In[14]:


def recommender(movie_name, data, model, n_recommendations):
    model.fit(data)
    idx= process.extractOne(movie_name, df_movies['title'])[2]
    print("Movie Selected: ", df_movies['title'][idx], 'Index : ', idx)
    print('Searching.....')
    distances, indices= model.kneighbors(data[idx], n_neighbors = n_recommendations)
    for i in indices:
        print(df_movies['title'][i].where(i!=idx))
        
recommender('Jurassic Park', mat_movies_user, model_knn, 20)





