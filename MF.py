import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds

data = pd.read_csv("TMDB_tv_dataset_v3.csv")

#Affichage des titres de colonnes
print("-------------------")
print(data.columns.tolist())
print("-------------------")
print("\n")

#Selection des colonnes importantes
columns = ['id', 'name', 'vote_average', 'popularity', 'number_of_episodes','number_of_seasons', 'genres', 'created_by', 'episode_run_time']
data = data[columns]
data = data.dropna(subset=['vote_average', 'popularity'])


# Exemple fictif d'interactions utilisateur-série
# Générer une matrice utilisateur-série aléatoire (à remplacer par de vraies interactions)
np.random.seed(42)
users = [f'user_{i}' for i in range(1, 101)]
series = data['id'].tolist()
ratings = {
    'user_id': np.random.choice(users, 1000),
    'series_id': np.random.choice(series, 1000),
    'rating': np.random.uniform(1, 5, 1000)
}
ratings_df = pd.DataFrame(ratings)

# Construire la matrice utilisateur-série
user_series_matrix = ratings_df.pivot(index='user_id', columns='series_id', values='rating').fillna(0)

# Factorisation de matrice avec SVD
matrix = user_series_matrix.values
user_mean = np.mean(matrix, axis=1)
matrix_demeaned = matrix - user_mean.reshape(-1, 1)

# Décomposition en valeurs singulières
U, sigma, Vt = svds(matrix_demeaned, k=50)
sigma = np.diag(sigma)

# Reconstruction de la matrice
reconstructed_matrix = np.dot(np.dot(U, sigma), Vt) + user_mean.reshape(-1, 1)
predicted_ratings = pd.DataFrame(reconstructed_matrix, index=user_series_matrix.index, columns=user_series_matrix.columns)

# Recommander des séries
user_id = 'user_1'  # Exemple d'utilisateur
user_predictions = predicted_ratings.loc[user_id].sort_values(ascending=False)

# Filtrer les séries non encore regardées
user_rated = user_series_matrix.loc[user_id]
recommendations = user_predictions[user_rated == 0].head(10)

# Afficher les recommandations
recommended_series = data[data['id'].isin(recommendations.index)]
print("Top 10 recommandations pour l'utilisateur", user_id)
print(recommended_series[['name', 'vote_average', 'popularity']])