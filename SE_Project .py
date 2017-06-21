# -*- coding: utf-8 -*-
"""
Created on Thu May 04 11:52:04 2017

@author: lily
"""               
import pandas as pd
import numpy as np

from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd
from sklearn import cross_validation as cv
from sklearn.metrics import mean_squared_error
from math import sqrt

from numpy.random import RandomState
from recommend.als import ALS
from recommend.utils.evaluation import RMSE


#Making Movie Recommendations
def SVD_recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.UserID == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'MovieID', right_on = 'MovieID').
                     sort_values(['Rating'], ascending=False)
                 )

    print 'User {0} has already rated {1} movies.'.format(userID, user_full.shape[0])
    print 'Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations)
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'MovieID',
               right_on = 'MovieID').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations

def SVD_not_recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=True)
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.UserID == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'MovieID', right_on = 'MovieID').
                     sort_values(['Rating'], ascending=True)
                 )

    print 'User {0} has already rated {1} movies.'.format(userID, user_full.shape[0])
    print 'Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations)
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    not_recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'MovieID',
               right_on = 'MovieID').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = True).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, not_recommendations


#RSME function
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

#Setting Up the Ratings Data
ratings_list = [i.strip().split("::") for i in open('/Users/user/Downloads/ml-1m/ratings.dat', 'r').readlines()]
users_list = [i.strip().split("::") for i in open('/Users/user/Downloads/ml-1m/users.dat', 'r').readlines()]
movies_list = [i.strip().split("::") for i in open('/Users/user/Downloads/ml-1m/movies.dat', 'r').readlines()]

ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

del ratings_df['Timestamp']

ratings = ratings_df.as_matrix()

movies_df.head()
ratings_df.head()

train_data, test_data = cv.train_test_split(ratings_df, test_size=0.25, random_state=0)

#Memory based content & collaborative filtering
#Create two user-item matrices, one for training and another for testing
train_df = train_data.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
train_matrix = train_df.as_matrix()

test_df = test_data.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
test_matrix = test_df.as_matrix()


#Singular Value Decomposition
U, sigma, Vt = svds(train_matrix, k = 10)
sigma = np.diag(sigma)
model_prediction1 = np.dot(np.dot(U, sigma), Vt)
print '-----------------svds-------------------- '
print 'RMSE: ' + str(rmse(model_prediction1, test_matrix))

model_preds_df1 = pd.DataFrame(model_prediction1, columns = train_df.columns)
print(model_preds_df1.head())

model_already_rated1, model_predictions1 = SVD_recommend_movies(model_preds_df1, 234, movies_df, ratings_df, 10)
print(model_already_rated1.Title.head(10))
print(model_predictions1.Title)

model_already_rated1, model_not_predictions1 = SVD_not_recommend_movies(model_preds_df1, 234, movies_df, ratings_df, 10)
print(model_not_predictions1.Title)
#--------------------------------------------------------------------------------#

#比svds的準確率好一點點
Ur, sigma, Vtr = randomized_svd(train_matrix, n_components=10, n_iter=10, random_state=None)
sigma = np.diag(sigma)

#Making Predictions from the Decomposed Matrices
model_prediction2 = np.dot(np.dot(Ur, sigma), Vtr)
print '-----------------randomized_svd-------------------- '
print 'RMSE: ' + str(rmse(model_prediction2, test_matrix))

model_preds_df2 = pd.DataFrame(model_prediction2, columns = train_df.columns)
print(model_preds_df2.head())

model_already_rated2, model_predictions2 = SVD_recommend_movies(model_preds_df2, 234, movies_df, ratings_df, 10)
print(model_already_rated2.Title.head(10))
print(model_predictions2.Title)

model_already_rated2, model_not_predictions2 = SVD_not_recommend_movies(model_preds_df2, 234, movies_df, ratings_df, 10)
print(model_not_predictions2.Title)
#--------------------------------------------------------------------------------#

rand_state = RandomState(0)

n_user = max(ratings[:, 0])
n_item = max(ratings[:, 1])

ratings[:, (0, 1)] -= 1


als_train_data, als_test_data = cv.train_test_split(ratings, test_size=0.25, random_state=0)


n_feature = 10
eval_iters = 10
print '-----------------ALS-------------------- '
print("n_user: %d, n_item: %d, n_feature: %d, training size: %d, test size: %d" % (
    n_user, n_item, n_feature, als_train_data.shape[0], als_test_data.shape[0]))
als = ALS(n_user=n_user, n_item=n_item, n_feature=n_feature,
          max_rating=5., min_rating=1., seed=0)

als.fit(als_train_data, n_iters=eval_iters)
train_preds = als.predict(als_train_data)
train_rmse = RMSE(train_preds, als_train_data[:, 2])
test_preds = als.predict(als_test_data)
test_rmse = RMSE(test_preds, als_test_data[:, 2])
print("after %d iterations, train RMSE: %.6f, test RMSE: %.6f" % \
      (eval_iters, train_rmse, test_rmse))


train_preds_rating = np.insert(als_train_data, 2, values=train_preds, axis=1)
train_preds_rating = np.delete(train_preds_rating, 3, 1)

train_preds_rating_list = train_preds_rating.tolist()

train_preds_rating_df = pd.DataFrame(train_preds_rating_list, columns = ['UserID', 'MovieID', 'Rating'], dtype = int)


def Non_Personalized_recommend_movies(predictions_df, movies_df, num_recommendations=5):

        grouped = predictions_df.groupby(by = 'MovieID', as_index=False).mean()
        del grouped['UserID']

        population_predictions = (pd.merge(grouped, movies_df, on = 'MovieID', 
                                          how = 'left').
                                          sort_values('Rating', ascending=False))
        
        popular_recommend = population_predictions.head(num_recommendations)
        
        return popular_recommend

def Non_Personalized_not_recommend_movies(predictions_df, movies_df, num_recommendations=5):

        grouped = predictions_df.groupby(by = 'MovieID', as_index=False).mean()
        del grouped['UserID']

        not_population_predictions = (pd.merge(grouped, movies_df, on = 'MovieID',
                                              how = 'left').
                                              sort_values('Rating', ascending=True))
        
        popular_not_recommend = not_population_predictions.head(num_recommendations)
        
        return popular_not_recommend

def CF_recommend_movies(predictions_df, theuser, movies_df, ratings_df, num_recommendations=5):
        
        user_data = ratings_df[ratings_df.UserID == (theuser)]
        user_full = (pd.merge(user_data, movies_df, how = 'left', on = 'MovieID').
                     sort_values('Rating', ascending=False))

        User_similarity = (pd.merge(predictions_df, user_full, how = 'right',
                                    on = 'MovieID'))
        del User_similarity['UserID_y']
        del User_similarity['Rating_y']

        User_similarity_pastRating = (User_similarity.groupby(by = 'UserID_x', 
                                      as_index=False).mean().
                                      merge(predictions_df, how = 'left', 
                                      left_on = 'UserID_x', right_on = 'UserID'))
        del User_similarity_pastRating['UserID_x']
        del User_similarity_pastRating['Rating_x']
        del User_similarity_pastRating['MovieID_x']

        User_similarity_pastMovie = (User_similarity_pastRating.groupby(by = 'MovieID_y', 
                                                     as_index=False).mean())
        del User_similarity_pastMovie['UserID']
        del User_similarity_pastMovie['Rating']


        user_full_movie = user_full.pop('MovieID')
        user_full_movie = pd.DataFrame(user_full_movie, columns = ['MovieID'], dtype = int)

        User_similarity_pastMovie.columns = ['MovieID']

        User_similarity_pastMovie = User_similarity_pastMovie.as_matrix().tolist()
        user_full_movie = user_full_movie.as_matrix().tolist()
        
        i = 0
        for i in range(len(user_full_movie)):
                User_similarity_pastMovie.remove(user_full_movie[i])
                
        User_similarity_pastMovie = pd.DataFrame(User_similarity_pastMovie, columns = ['MovieID'])

        User_similarity_recommend = (pd.merge(User_similarity_pastMovie, User_similarity_pastRating, 
                                              how = 'left', left_on = 'MovieID', 
                                              right_on = 'MovieID_y'))

        del User_similarity_recommend['MovieID']
        User_similarity_recommend.columns = ['UserID', 'MovieID', 'Rating']
        User_similarity_recommend = (User_similarity_recommend.groupby(by = 'MovieID', 
                                     as_index=False).mean().
                                     sort_values('Rating', ascending=False))
        del User_similarity_recommend['UserID']

        User_similarity_recommend = (pd.merge(User_similarity_recommend, movies_df, 
                                              how = 'left', on = 'MovieID'))
        
        CF_recommend = User_similarity_recommend.head(num_recommendations)
        
        return CF_recommend

def CF_not_recommend_movies(predictions_df, theuser, movies_df, ratings_df, num_recommendations=5):
        
        user_data = ratings_df[ratings_df.UserID == (theuser)]
        user_full = (pd.merge(user_data, movies_df, how = 'left', on = 'MovieID').
                     sort_values('Rating', ascending=False))

        User_similarity = (pd.merge(predictions_df, user_full, how = 'right',
                                    on = 'MovieID'))
        del User_similarity['UserID_y']
        del User_similarity['Rating_y']

        User_similarity_pastRating = (User_similarity.groupby(by = 'UserID_x', 
                                      as_index=False).mean().
                                      merge(predictions_df, how = 'left', 
                                      left_on = 'UserID_x', right_on = 'UserID'))
        del User_similarity_pastRating['UserID_x']
        del User_similarity_pastRating['Rating_x']
        del User_similarity_pastRating['MovieID_x']

        User_similarity_pastMovie = (User_similarity_pastRating.groupby(by = 'MovieID_y', 
                                                     as_index=False).mean())
        del User_similarity_pastMovie['UserID']
        del User_similarity_pastMovie['Rating']


        user_full_movie = user_full.pop('MovieID')
        user_full_movie = pd.DataFrame(user_full_movie, columns = ['MovieID'], dtype = int)

        User_similarity_pastMovie.columns = ['MovieID']

        User_similarity_pastMovie = User_similarity_pastMovie.as_matrix().tolist()
        user_full_movie = user_full_movie.as_matrix().tolist()
        
        i = 0
        for i in range(len(user_full_movie)):
                User_similarity_pastMovie.remove(user_full_movie[i])
                
        User_similarity_pastMovie = pd.DataFrame(User_similarity_pastMovie, columns = ['MovieID'])

        User_similarity_recommend = (pd.merge(User_similarity_pastMovie, User_similarity_pastRating, 
                                              how = 'left', left_on = 'MovieID', 
                                              right_on = 'MovieID_y'))

        del User_similarity_recommend['MovieID']
        User_similarity_recommend.columns = ['UserID', 'MovieID', 'Rating']
        User_similarity_recommend = (User_similarity_recommend.groupby(by = 'MovieID', 
                                     as_index=False).mean().
                                     sort_values('Rating', ascending=True))
        del User_similarity_recommend['UserID']

        User_similarity_recommend = (pd.merge(User_similarity_recommend, movies_df, 
                                              how = 'left', on = 'MovieID'))
        
        CF_not_recommend = User_similarity_recommend.head(num_recommendations)
        
        return CF_not_recommend


Non_Personalized_recommend = Non_Personalized_recommend_movies(train_preds_rating_df, movies_df, 10)
print 'Non_Personalized_recommend: '
print(Non_Personalized_recommend.Title)

Non_Personalized_not_recommend = Non_Personalized_not_recommend_movies(train_preds_rating_df, movies_df, 10)
print 'Non_Personalized_not_recommend: '
print(Non_Personalized_not_recommend.Title)

CF_recommend = CF_recommend_movies(train_preds_rating_df, 234, movies_df, ratings_df, 10)
print 'CF_recommend: '
print(CF_recommend.Title)

CF_not_recommend = CF_not_recommend_movies(train_preds_rating_df, 234, movies_df, ratings_df, 10)
print 'CF_not_recommend: '
print(CF_not_recommend.Title)

