import pandas as pd
from pandas import Series, DataFrame
import scipy.sparse
from scipy.sparse import lil_matrix, coo_matrix, linalg
import numpy as np
import sys
from sklearn.metrics import pairwise

from util import Timer

review_file = "tinyset.npz"
user_file = "tinyusers_df.json"
movie_file = "tinymovies_df.json"
movie_concept_file = "tinyMovie_to_Concept.npy"

try:
    movie_reviews = scipy.sparse.load_npz(review_file)
    movie_reviews = movie_reviews.tolil().astype(np.int8)
    users = pd.read_json(user_file, lines=True)
    movies = pd.read_json(movie_file, lines=True)
    movie_to_concept = np.load(movie_concept_file)
except Exception as e:
    raise

# Load in a json file and use it to populate a sparse matrix with reviewrs as rows and movies as columns
# saves the resultant output to an npz file so we can retrieve it later without having to re-create it.
def Create_NPZ(file, output_file):
    reviews = pd.read_json(file, lines=True)
    reviews.sort_values(['reviewerID', 'asin'], ascending=[True, True], inplace=True)

    # create two DataFrames to act as indexes for the matrix
    # the index of the movie dataframe is the column number
    # the index of the users dataframe is the row number
    # example:
    #    users[users.userID=="A01174011QPNX7GZF4B92"].index.values[0] returns 7
    #    movies[movies.asin=="6300248135"].index.values[0] returns 9
    #    the value of m_reviews[7,9] is 5
    movies = DataFrame(data=reviews.asin.unique(), columns=["asin"])
    users = DataFrame(data=reviews.reviewerID.unique(), columns=["userID"])

    # initialize a new lil matrix, with size (users,movies)
    m_reviews = lil_matrix((len(users), len(movies)), dtype=np.int8)

    t = Timer()
    t.Start()
    count = 0
    total = reviews.shape[0]
    for row in reviews.iterrows():
        count += 1
        m_row_ID = users[users.userID==row[1].reviewerID].index.values[0]
        m_col_ID = movies[movies.asin==row[1].asin].index.values[0]
        m_value = row[1].overall
        m_reviews[m_row_ID, m_col_ID] = m_value

        # just so I know it's still working
        if count % 1000 == 0:
            sys.stdout.write("{} of {} ({} remaining)...\n".format(count, total, (total-count)))
            sys.stdout.flush()

    # while we're here, lets just compute the average movie scores and save them in that dataframe
    for row in range(m_reviews.shape[0]):
        movies.at[row,"avg_rating"] = m_reviews[row].sum()/m_reviews[row].nnz
        if row % 1000 == 0:
            sys.stdout.write("Getting Average Movie Score: {} of {}...\n".format(row, m_reviews.shape[0]))
            sys.stdout.flush()

    # save the dataframes to files, because I've lost them twice already
    movies.to_json(output_file+"movies_df.json",orient='records', lines=True)
    users.to_json(output_file+"users_df.json",orient='records', lines=True)

    # convert to a coo matrix so we can save it
    m_reviews = coo_matrix(m_reviews)
    t.Stop()
    print("Completed in ",t)

    # save to file
    scipy.sparse.save_npz(output_file + ".npz", m_reviews)

def Create_SVD(min_energy=0.8):
    # figure out how much we can reduce the review matrix
    k=min(movie_reviews.shape)-1
    U,s,V = linalg.svds(movie_reviews.asfptype(), k=k)
    total_energy = np.square(s).sum()

    energy = total_energy
    while energy > (total_energy*min_energy):
        k -= 1
        s = np.delete(s,0)
        energy = np.square(s).sum()
        print("Energy of SVD Decomp k={}: {:.3f}".format(k,energy/total_energy))
    print("Making SVD with k =",k)
    U,s,V = linalg.svds(movie_reviews.asfptype(), k=k)
    np.save(movie_concept_file,V)

def Get_Similar_Users(UserID, N=5, min_sim=0.8):
    user_index = users[users.user_id==UserID].index.values[0]
    this_user_reviews = movie_reviews[user_index].toarray()[0]
    this_user_vector = np.sum(this_user_reviews*movie_to_concept, axis=1)
    user_list = []
    for row in range(movie_reviews.shape[0]):
        if row == user_index: # skip this user's own row
            continue
        user_reviews = movie_reviews[row].toarray()[0]
        user_vector = np.sum(user_reviews*movie_to_concept, axis=1)
        similarity = pairwise.cosine_similarity([this_user_vector], [user_vector])[0][0]
        if similarity >= min_sim:
            user_list.append({"id":row,"sim":similarity})
    user_list = DataFrame(user_list).sort_values("sim", ascending=False).head(N)
    print(user_list)
    return user_list

if __name__=="__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "create":
            if sys.argv[2] == "dev":
                print("creating dev file...")
                Create_NPZ("reviews.dev.json", "devset")
            elif sys.argv[2] == "training":
                print("creating training file...")
                Create_NPZ("reviews.training.json", "trainingset")
            elif sys.argv[2] == "SVD":
                Create_SVD()
        elif sys.argv[1] == "similar":
            Get_Similar_Users(sys.argv[2])
