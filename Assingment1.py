import pandas as pd
from pandas import Series, DataFrame
import scipy.sparse
from scipy.sparse import lil_matrix, coo_matrix, linalg
import numpy as np
import sys

from util import Timer

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
if __name__=="__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "create":
            if sys.argv[2] == "dev":
                print("creating dev file...")
                Create_NPZ("reviews.dev.json", "devset")
            elif sys.argv[2] == "training":
                print("creating training file...")
                Create_NPZ("reviews.training.json", "trainingset")
