import pandas as pd
from pandas import Series, DataFrame
import scipy.sparse
from scipy.sparse import lil_matrix, coo_matrix, linalg
import numpy as np
import sys
from sklearn.metrics import pairwise

from util import Timer, super_print

review_file = "devset.npz"
user_file = "devsetusers_df.json"
movie_file = "devsetmovies_df.json"
movie_concept_file = "tinyMovie_to_Concept.npy"

try:
    movie_reviews = scipy.sparse.load_npz(review_file)
    movie_reviews = movie_reviews.tolil().astype(np.int8)
    users = pd.read_json(user_file, lines=True)
    movies = pd.read_json(movie_file, lines=True)
    movie_to_concept = np.load(movie_concept_file)
except Exception as e:
    print("Unable to load all files...")

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

def Create_SVD(k, min_energy=0.8):
    # figure out how much we can reduce the review matrix
    # k=min(movie_reviews.shape)-1
    # U,s,V = linalg.svds(movie_reviews.asfptype(), k=k)
    # total_energy = np.square(s).sum()
    #
    # energy = total_energy
    # while energy > (total_energy*min_energy):
    #     k -= 1
    #     s = np.delete(s,0)
    #     energy = np.square(s).sum()
    #     print("Energy of SVD Decomp k={}: {:.3f}".format(k,energy/total_energy))
    # print("Making SVD with k =",k)
    # U,s,V = linalg.svds(movie_reviews.asfptype(), k=k)
    # np.save(movie_concept_file,V)
    super_print("Complete")

def Get_Indexes_With_Reviews(Movie):
    movie_index = movies[movies.asin==Movie].index[0]
    return movie_reviews.getcol(movie_index).nonzero()[0]

def Get_Similar_Users(UserID, ids_to_check=[], N=5, min_sim=0.8):
    user_index = users[users.userID==UserID].index.values[0]
    this_user_reviews = movie_reviews[user_index].toarray()[0]
    this_user_vector = np.sum(this_user_reviews*movie_to_concept, axis=1)
    super_print("Searching for Users similar to " + UserID)
    if ids_to_check == []:
        ids_to_check = range(m_reviews.shape[0])
    user_list = []
    for row in ids_to_check:
        if row == user_index: # skip this user's own row
            continue
        user_reviews = movie_reviews[row].toarray()[0]
        user_vector = np.sum(user_reviews*movie_to_concept, axis=1)
        similarity = pairwise.cosine_similarity([this_user_vector], [user_vector])[0][0]
        user_list.append({"id":row,"sim":similarity})
        if similarity >= min_sim:
            super_print("Found User with similarity {}".format(similarity))
        else:
            super_print("(Poor User similarity: {:.2f})".format(similarity))
    user_list = DataFrame(user_list).sort_values("sim", ascending=False).head(N)
    print(user_list)
    return user_list

def PredictReview(userID, movieID):
    # parameters
    N = 10 # what's the maximum number of similar users to consider

    # We're only interested in checking other users who have also reviewed this movie
    indexes_to_check = Get_Indexes_With_Reviews(movieID)

    # unless I come up with something better, return the average rating for a movie that's never been rated.
    if indexes_to_check == []:
        return 4.110994929404886

    # Get our indexes so we can find them on the review matrix
    user_index = users[users.userID==userID].index.values[0]
    movie_index = movies[movies.asin==movieID].index[0]

    # we need get this user's review vector to compare to all the other users
    # get this by taking their reviews and multiplying by the concepts matrix
    this_user_reviews = movie_reviews[user_index].toarray()[0]
    this_user_vector = np.sum(this_user_reviews*movie_to_concept, axis=1)

    user_list = []
    for row in indexes_to_check:
        user_reviews = movie_reviews[row].toarray()[0]
        user_vector = np.sum(user_reviews*movie_to_concept, axis=1)
        similarity = pairwise.cosine_similarity([this_user_vector], [user_vector])[0][0]
        user_rating = user_reviews[movie_index]
        if similarity > 0:
            user_list.append({"id":row,"sim":similarity,"rating":user_rating})

    user_list = DataFrame(user_list).sort_values("sim", ascending=False).head(N) # sort by similarity descending, limit by N
    user_list["weighted_rating"] = user_list.rating * user_list.sim # get ratings weighted by similarity
    predicted_rating = user_list.weighted_rating.sum()/user_list.sim.sum() # predicted rating is the weighted average of the similar users by similarity
    return predicted_rating

def GetPredictions():
    results = []
    count = 0
    total_count = req_reviews.shape[0]
    for row in req_reviews.iterrows():
        count += 1
        t = Timer()
        t.Start()
        predicted = PredictReview(row[1].reviewerID, row[1].asin)
        results.append({"datapointID":row[1].datapointID,"overall":predicted})
        t.Stop()
        super_print("({} of {}) Review for [{},{}]={:.2f} ({:.2f}s)".format(count, total_count,row[1].reviewerID, row[1].asin,predicted,t.elapsed))
    DataFrame(results).to_csv("output.csv", index=False)

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
                Create_SVD(1000)
        elif sys.argv[1] == "similar":
            Get_Similar_Users(sys.argv[2])
        elif sys.argv[1] == "predict":
            if sys.argv[2] == "all":
                GetPredictions()
            else:
                PredictReview(sys.argv[2],sys.argv[3])
