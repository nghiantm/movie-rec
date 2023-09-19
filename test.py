import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import seaborn as sns

import pickle

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

if __name__ == '__main__':

	movies = pd.read_csv("movies.csv")

	with open('final_dataset.pickle', 'rb') as f:
		final_dataset = pickle.load(f)

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

	user_input = str(input("Enter a movie you like: "))
	suggestdf = get_movie_recommendation(user_input)
	print(suggestdf)