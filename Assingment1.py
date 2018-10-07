import pandas as pd
from pandas import Series, DataFrame
import scipy.sparse
from scipy.sparse import lil_matrix, coo_matrix, linalg
import numpy as np
import sys
from sklearn.metrics import pairwise

from util import Timer, super_print

review_file = "1_trainingset.npz"
user_file = "trainingsetusers_df.json"
movie_file = "trainingsetmovies_df.json"
movie_concept_file = "trainingMovie_to_Concept_100.npy"
req_reviews_file = "reviews.test.unlabeled.csv"

try:
    movie_reviews = scipy.sparse.load_npz(review_file)
    movie_reviews = movie_reviews.tolil().astype(np.int8)
    users = pd.read_json(user_file, lines=True)
    movies = pd.read_json(movie_file, lines=True)
    movie_to_concept = np.load(movie_concept_file)
    req_reviews = pd.read_csv(req_reviews_file)
    user_vectors = np.load(user_vectors_file)
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
    # print(s)
    #
    # energy = total_energy
    # while energy > (total_energy*min_energy):
    #     k -= 1
    #     s = np.delete(s,0)
    #     energy = np.square(s).sum()
    #     print("Energy of SVD Decomp k={}: {:.3f}".format(k,energy/total_energy))
    super_print("Making SVD with k = " + str(k))
    user_to_concept,s,movie_to_concept = linalg.svds(movie_reviews.asfptype(), k=k)
    np.save(movie_concept_file,movie_to_concept)
    super_print("Complete")

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

def old_PredictReview(userID, movieID):
    # parameters
    N = 20 # what's the maximum number of similar users to consider
    min_sim = 0# what is the minimum similarity we're going to consider

    # We're only interested in checking other users who have also reviewed this movie
    indexes_to_check = Get_Indexes_With_Reviews(movieID)

    # unless I come up with something better, return the average rating for a movie that's never been rated.
    if indexes_to_check == []:
        return 4.110994929404886

    try:
        # Get our indexes so we can find them on the review matrix
        user_index = users[users.userID==userID].index.values[0]
        movie_index = movies[movies.asin==movieID].index[0]

        this_user_vector = user_vectors[user_index]

        user_list = []
        for row in indexes_to_check:

            user_vector = user_vectors[row]
            similarity = pairwise.cosine_similarity([this_user_vector], [user_vector])[0][0]

            if similarity > min_sim:
                user_reviews = movie_reviews[row].toarray()[0]
                user_list.append({"id":row,"sim":similarity,"rating":user_reviews[movie_index]})

        user_list = DataFrame(user_list).sort_values("sim", ascending=False).head(N) # sort by similarity descending, limit by N
        user_list["weighted_rating"] = user_list.rating * user_list.sim # get ratings weighted by similarity
        predicted_rating = user_list.weighted_rating.sum()/user_list.sim.sum() # predicted rating is the weighted average of the similar users by similarity
        return predicted_rating

    except:
        print("oh god I have no idea") # if all else fails return the average
        return 4.110994929404886

def PredictReview(userID, movieID):
    try:
        # parameters
        N = 20 # how many movies to consider when we're weighting
        min_sim = 0.2 # what is the minimum level of similarity to consider

        # Get our indexes so we can find them on the review matrix
        user_index = users[users.userID==userID].index.values[0]
        movie_index = movies[movies.asin==movieID].index[0]

        # get the movie->concept matrix for this movie
        this_movie_concept_vector = movie_to_concept[:,movie_index]

        # get a list of the indexs of movies this user has reviewed
        user_reviews = movie_reviews.getrow(user_index).nonzero()[1]

        # check how similar each movie is to the one we're trying to guess
        movie_list = []
        for m_id in user_reviews:
            # get the concept vector for this movie
            m_concept_vector = movie_to_concept[:,m_id]
            # calculate how similar this movie's ratings are
            similarity = pairwise.cosine_similarity([this_movie_concept_vector], [m_concept_vector])[0][0]
            # add it to the list so we can sort it
            if similarity > min_sim:
                movie_list.append({"id":m_id,"sim":similarity,"rating":movie_reviews[user_index,m_id]})
        # sort the list by similarity
        movie_list = DataFrame(movie_list).sort_values("sim", ascending=False).head(N)
        movie_list["weighted"] = movie_list.rating*movie_list.sim
        predicted_rating = movie_list.weighted.sum()/movie_list.sim.sum()
        return predicted_rating
    except:
        # if anything at all goes wrong, just spit out the average review score
        return 4.110994929404886

def GetPredictions(file=None):
    results = []
    records_to_skip = 0
    if file is not None:
        results = pd.read_csv(file).to_dict('records')
        records_to_skip = len(results)
    count = 0
    total_count = req_reviews.shape[0]
    t = Timer()
    t.Start()
    for row in req_reviews.iterrows():
        count += 1
        if count > records_to_skip:
            predicted = PredictReview(row[1].reviewerID, row[1].asin)
            # super_print("{}: {}".format(count, predicted))
            results.append({"datapointID":row[1].datapointID,"overall":predicted})
            if count % 100 == 0:
                t.Stop()
                super_print("({} of {}) ({:.2f}s/prediction)".format(count, total_count, t.elapsed/100))
                t.Start()
                DataFrame(results).to_csv("output.csv", index=False)
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
                Create_SVD(10)
        elif sys.argv[1] == "similar":
            Get_Similar_Users(sys.argv[2])
        elif sys.argv[1] == "predict":
            if sys.argv[2] == "all":
                GetPredictions()
            else:
                print(PredictReview(sys.argv[2],sys.argv[3]))
