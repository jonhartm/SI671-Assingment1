import pandas as pd
from pandas import Series, DataFrame
import scipy.sparse
from scipy.sparse import lil_matrix, coo_matrix, linalg
import numpy as np
import sys
from sklearn.metrics import pairwise

from util import Timer, super_print

review_file = "1_trainingset.npz"
user_file = "allusersbaselines_df.json"
movie_file = "allmoviesbaselines_df.json"
movie_concept_file = "trainingMovie_to_Concept_100.npy"
req_reviews_file = "reviews.test.unlabeled.csv"

try:
    # load in the npz user-item matrix
    movie_reviews = scipy.sparse.load_npz(review_file)
    movie_reviews = movie_reviews.tocsr().astype(np.int8) # convert it to a lil_matrix with int8 (1 <= values <= 5)
    users = pd.read_json(user_file, lines=True) # get the dictionary that gives me the user->user index values
    movies = pd.read_json(movie_file, lines=True) # get the dictionary that gives me the movie->movie index values
    movie_to_concept = np.load(movie_concept_file) # get teh movie-to-concept matrix from the SVD
    req_reviews = pd.read_csv(req_reviews_file) # get the list of reviews to find
except Exception as e:
    # let me know if any one of those isn't present.
    print("Unable to load all files...")

average_review_score = 4.110994929404886 # straight average of all reviews in the matrix

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
    # iterate through all rows in the reviews file I've loaded in
    for row in reviews.iterrows():
        count += 1
        # grab the user and movie IDs from the dictionaries I made
        m_row_ID = users[users.userID==row[1].reviewerID].index.values[0]
        m_col_ID = movies[movies.asin==row[1].asin].index.values[0]
        m_value = row[1].overall
        # assign the rating value to the matrix coordinate [user,movie]
        m_reviews[m_row_ID, m_col_ID] = m_value

        # just so I know it's still working
        if count % 1000 == 0:
            sys.stdout.write("{} of {} ({} remaining)...\n".format(count, total, (total-count)))
            sys.stdout.flush()

    # save the user->userid and movie->movieid dictionaries to files, because I've lost them twice already
    movies.to_json(output_file+"movies_df.json",orient='records', lines=True)
    users.to_json(output_file+"users_df.json",orient='records', lines=True)

    # convert to a coo matrix so we can save it
    m_reviews = coo_matrix(m_reviews)
    # save to file
    scipy.sparse.save_npz(output_file + ".npz", m_reviews)
    t.Stop()
    print("Completed in ",t)


# Create an SVD matrix from the movie reviews user-item matrix
# saves the movie-to-concept matrix to disk so we can get it later
def Create_SVD(k):
    super_print("Making SVD with k = " + str(k))
    user_to_concept,s,movie_to_concept = linalg.svds(movie_reviews.asfptype(), k=k)
    np.save(movie_concept_file,movie_to_concept)
    super_print("Complete")

# get the average rating of all this user's reviews
def get_user_baseline_reviews(user):
    super_print("user "+str(user.name))
    total = 0
    count = 0
    for m_id in movie_reviews[user.name].nonzero()[1]:
        count += 1
        total += movie_reviews[user.name,m_id]
    return total/count - average_review_score

# get the averate rating all users have given this movie
def get_movie_baseline_reviews(movie):
    super_print("movie "+str(movie.name))
    total = 0
    count = 0
    for u_id in movie_reviews[:,movie.name].nonzero()[0]:
        count += 1
        total += movie_reviews[u_id,movie.name]
    return total/count - average_review_score

# applies the above two functions over the users and movies DataFrames
# saves the output to files so we only have to do this once
def Get_Baseline_Reviews():
    users["baseline"] = users.apply(get_user_baseline_reviews, axis=1)
    users.to_json("allusersbaselines_df.json",orient='records', lines=True)

    movies["baseline"] = movies.apply(get_movie_baseline_reviews, axis=1)
    movies.to_json("allmoviesbaselines_df.json",orient='records', lines=True)

# Given a userID and movieID, predict the rating that user would give to that movie
def PredictReview(
                    userID,
                    movieID,
                    min_sim=0.75,
                    baseline_weighting=0.75,
                    desired_sim_records=2,
                    backup_min_sim = 0.5,
                    debug=False
                ):

    # just make sure - I made this mistake once. 
    if backup_min_sim > min_sim:
        raise Exception("min_sim must be greater than backup_min_sim")

    # Get our indexes so we can find them on the review matrix
    user_row = users[users.userID==userID]
    movie_row = movies[movies.asin==movieID]

    # cold start problem
    if len(user_row)==0 and len(movie_row)!=0: # new user, but not a new movie
        if (debug): print("New User")
        return average_review_score+movie_row.baseline.values[0]

    if len(user_row)!=0 and len(movie_row)==0: # not a new user, but a new movie
        if (debug): print("New Movie")
        return average_review_score+user_row.baseline.values[0]

    if len(user_row)==0 and len(movie_row)==0: # new user and new movie
        if (debug): print("New User and New Movie")
        return average_review_score


    user_index = user_row.index[0]
    movie_index = movie_row.index[0]

    user_baseline = user_row.baseline.values[0] * baseline_weighting
    movie_baseline = movie_row.baseline.values[0] * baseline_weighting

    if (debug): print("user/movie baselines:",user_baseline,movie_baseline)

    # get the movie->concept matrix for this movie
    this_movie_concept_vector = movie_to_concept[:,movie_index]

    # get a list of the indexs of movies this user has reviewed
    user_reviews = movie_reviews.getrow(user_index).nonzero()[1]

    # check how similar each movie is to the one we're trying to guess
    movie_list = []
    backup_movie_list = []
    if (debug): print("found",len(user_reviews),"other reviews")
    for m_id in user_reviews:
        # get the concept vector for this movie
        m_concept_vector = movie_to_concept[:,m_id]
        # calculate how similar this movie's ratings are
        similarity = pairwise.cosine_similarity([this_movie_concept_vector], [m_concept_vector])[0][0]
        # add it to the list so we can sort it
        if similarity > backup_min_sim:
            # get this movie's baseline
            this_movie_baseline = movies.iloc[m_id].baseline
            rating = movie_reviews[user_index,m_id]
            weighted_similarity = (similarity*(rating-this_movie_baseline))
            backup_movie_list.append([similarity,weighted_similarity])
            if similarity > min_sim:
                movie_list.append([similarity,weighted_similarity]) # add a entry [sim,weighted rating]
                if (debug): print("movie: {} rated {} - sim: {:.4f} - weighted-sim: {:.4f}".format(m_id,rating,similarity,weighted_similarity))
        else:
            if (debug): print("poor sim: {:.4f}".format(similarity))

    # if we don't have the desired minimum similar movies, use the backup list with it's lower tolerance instead
    if len(movie_list) > desired_sim_records:
        movie_list = backup_movie_list
    # sort the list by similarity
    movie_list = np.array(movie_list) # convert to a numpy array\
    movie_list[::-1].sort(0) # sort the array by similarity descinding
    if len(movie_list) == 0: # we didn't find any similar movies
        rating = average_review_score + user_baseline + movie_baseline
    else:

        movie_list = np.sum(movie_list,axis=0) # compress the array into the sum of it's columns
        if (debug): print("summed cols",movie_list)
        if (debug): print("weighted_review",(movie_list[1]/movie_list[0]))
        if (debug): print("baselines",user_baseline,movie_baseline,"total",(user_baseline+movie_baseline))
        rating = user_baseline + movie_baseline + (movie_list[1]/movie_list[0]) # return the baseline + the weighted review average

    # no ratings above 5 or below 1
    rating = max(1, rating)
    rating = min(5, rating)

    if (debug): print("RATING:",rating)
    return rating

# Roll through reviews.test.unlabeled.csv and get a predicted review for each user/movie combination
# saves the results to "output.csv" (also saves every 1000 predictions, just in case)
def GetPredictions(file=None):
    results = []

    # So I don't have to re-do a lot of predictions if I start and stop
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
        if count > records_to_skip: # in the event we're continuing a file, jump to the last record
            predicted = PredictReview(row[1].reviewerID, row[1].asin)
            results.append({"datapointID":row[1].datapointID,"overall":predicted})
            if count % 1000 == 0:
                # informative prints so we know it's still working
                t.Stop()
                super_print("({} of {}) ({:.4f}s/prediction)".format(count, total_count, t.elapsed/1000))
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
                Create_SVD(int(sys.argv[3]))
            elif sys.argv[2] == "baselines":
                Get_Baseline_Reviews()
        elif sys.argv[1] == "similar":
            Get_Similar_Users(sys.argv[2])
        elif sys.argv[1] == "predict":
            if sys.argv[2] == "all":
                GetPredictions()
            elif sys.argv[2] == "continue":
                GetPredictions("output.csv")
            else:
                print(PredictReview(sys.argv[2],sys.argv[3]))
