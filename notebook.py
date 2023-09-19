# %% [markdown]
# 
# 
# # Item based filtering
# This approach is mostly preferred since the movie don't change much. We can rerun this model once a week unlike User based where we have to frequently run the model.
# 
# In this kernel, We look at the implementation of Item based filtering

# %%
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import seaborn as sns

import sys

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# %

# %% [markdown]
# Ratings dataset has 
# * userId - unique for each user
# * movieId - using this feature ,we take the title of the movie from movies dataset
# * rating - Ratings given by each user to all the movies using this I'm are going to predict the top 10 similar movies

# %%

# %% [markdown]
# Movie dataset has 
# * movieId - once the recommendation is done, we get list of all similar movieId and get the title for each movie from this dataset. 
# * genres -  which is not required for this filtering approach

# %%
ratings = ratings[ratings["userId"].isin(np.arange(1,10001))]

# %%
final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')

# %%
final_dataset.fillna(0,inplace=True)

# %% [markdown]
# In a real world, ratings are very sparse and data points are mostly collected from very popular movies and highly engaged users. So we will reduce the noise by adding some filters and qualify the movies for the final dataset.
# * To qualify a movie, minimum 10 users should have voted a movie.
# * To qualify a user, minimum 50 movies should have voted by the user.
# 

# %%
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

# %%
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

f,ax = plt.subplots(1,1,figsize=(16,4))
# ratings['rating'].plot(kind='hist')
plt.scatter(no_user_voted.index,no_user_voted,color='mediumseagreen')
plt.axhline(y=10,color='r')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.show()

# %%
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]

# %%
f,ax = plt.subplots(1,1,figsize=(16,4))
plt.scatter(no_movies_voted.index,no_movies_voted,color='mediumseagreen')
plt.axhline(y=50,color='r')
plt.xlabel('UserId')
plt.ylabel('No. of votes by user')
plt.show()

# %%
final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]
final_dataset.to_csv('final_dataset.csv')

# %% [markdown]
# Our final_dataset has dimensions of **2121 * 378** where most of the values are sparse. I took only small dataset but for
# original large dataset of movie lens which has more than **100000** features, this will sure hang our system when this has 
# feed to model. To reduce the sparsity we use csr_matric scipy lib. I'll give an example how it works

# %%
sample = np.array([[0,0,3,0,0],[4,0,0,0,2],[0,0,0,0,1]])
sparsity = 1.0 - ( np.count_nonzero(sample) / float(sample.size) )
print(sparsity)

# %%
csr_sample = csr_matrix(sample)
print(csr_sample)

# %% [markdown]
# * As you can see there is no sparse value in the csr_sample and values are assigned as rows and column index. for the 0th row and 2nd column , value is 3 . Look at the original dataset where the values at the right place. This is how it works using todense method you can take it back to original dataset.
# * Most of the sklearn works with sparse matrix. surely this will improve our performance

# %%
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

# %% [markdown]
# We use cosine distance metric which is very fast and preferable than pearson coefficient. Please don't use euclidean distance which will not work when the values are equidistant.

# %%
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

# %%
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# %%
def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),\
                               key=lambda x: x[1])[:0:-1]
        
        recommend_frame = []
        
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return df
    
    else:
        
        return "No movies found. Please check your input"

# %%
get_movie_recommendation('Inception')

# %%
get_movie_recommendation('Memento')

# %%
get_movie_recommendation('Interstellar')

# %% [markdown]
# Our model works perfectly predicting the recommendation based on user behaviour and past search. So we conclude our 
# collaborative filtering here.
# 


