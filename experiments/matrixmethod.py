import numpy as np

from scipy import spatial
from sklearn.cluster import DBSCAN


centroids = [2.60999494,16.62854731,30.21092294,44.44653696,61.54627043,91.75547689,134.94938152,195.94670594, 336.74164796,527.55770016,775.6417704, 100000]
CENTROIDS = centroids


def _traverse_matrix(clust_mat,
                     padding_item='11D',
                     llists=[],
                     unvisited_rows=[]):
    """Runs pathbuilding for all clusters and returns a list of lists with paths."""
    for c in np.unique(clust_mat):
        # We dont explore for this cluster_id (padding)
        if isinstance(padding_item, str):
            if c in padding_item:
                continue
        if padding_item in c:
            continue

        # This list accumulates all the subseries for each one of the clusters
        lisst = []

        # If we consumed all the available rows on previous rounds, we stop
        if len(unvisited_rows) == 0:
            break

        row_i = unvisited_rows[0]  # 0

        reached_end = False

        while len(unvisited_rows) > 0:

            row_i = min(row_i, clust_mat.shape[1] - 1)

            # Search on the current row all the columns which have a match: (i,j) == custer_id
            idx_match_serie = (np.argwhere(clust_mat[row_i, row_i:] == c) + row_i).flatten()

            # We can only connect rows that haven't yet been connected
            idx_match_serie = [i for i in idx_match_serie.flatten().tolist() if
                               i in unvisited_rows]  # or i >= final_clust_mat.shape[0]-1

            # If no matches in this row, we stop the search (strict behavior)
            if len(idx_match_serie) == 0:
                break
                # row_i +=1
                # if row_i >= clust_mat.shape[0]:
                #    break
                # continue

            # Take the first available row to connect
            idx_match_serie = idx_match_serie[0]

            current_elem = clust_mat[row_i, idx_match_serie]

            # If we have a match, we connect the current row to the previous ones and also the next row (i.e, the row that is equal to the column that matched)
            if current_elem == c:
                # Connect current row and next point
                lisst.append(row_i)
                lisst.append(idx_match_serie)
                #print(f"CID: {c} - Appending {row_i} - {idx_match_serie}")

                # Remove them from the "available to connect" list
                if row_i in unvisited_rows:
                    unvisited_rows.remove(row_i)
                if idx_match_serie in unvisited_rows:
                    unvisited_rows.remove(idx_match_serie)

                # Move the pointer to the next row!
                row_i = idx_match_serie

                if row_i >= clust_mat.shape[0] - 1:
                    if reached_end:  # previously reached end
                        break

                    # In case we reached the end of the matrix, we can traverse the matrix again in case certain rows can be connected again.
                    # If this isn't run, the built paths can be incomplete.
                    # Note this second traversal is faster as only few items will remain unvisited here.
                    reached_end = True
                    if len(unvisited_rows) == 0:
                        break
                    row_i = unvisited_rows[0]
                    continue

        t = list(set(sorted(lisst)))  # TODO: Mirar ñapa
        if len(t) > 3:
            llists.append(t)
        else:  # rollback!
            for i in t:
                unvisited_rows.append(i)
            unvisited_rows.sort()


def optim_paths(clust_mat, padding_item='11D'):
    llists = []
    unvisited_rows = [i for i in range(clust_mat.shape[0])]

    for _ in range(2):
        # Now, Cluster-IDs don't follow any order, so some of them can contain paths only visible after other clusters have been explored
        # Repeating the search would fix this issue. (Note that when there aren't more unvisited rows, the loop will stop).
        _traverse_matrix(clust_mat, padding_item, llists, unvisited_rows)

    llists = [sorted(l) for l in llists]  # Ensure output paths are ordered

    return llists


def matrix_optim(difsarr, mask_zeros=True):
    """Computes the pairwise distances of days, given the array of sequencial differences (as ints)"""
    d = np.concatenate(([0], difsarr.cumsum()))[None, :]  #  original Vero's behavior with duplicated first entry
    #d = difsarr.cumsum()[None, :]   # Rework
    dists = spatial.distance.pdist(d.T, metric='cityblock')   # TODO: sklearn has the pairwise distance, which supports parallelization (idk if there is any speedup)
    mat = spatial.distance.squareform(dists)
    mat = np.triu(mat)
    #mat = np.tril(mat)

    if mask_zeros:
        mat[np.where(mat==0)]=10000000

    return mat


def diff_imp_matrix_optim(amoutns):
    """ Computes the percent difference matrix of amounts (given the full array of amounts)"""
    diff_imps = np.array(amoutns)
    mat = spatial.distance.squareform(
        spatial.distance.pdist(diff_imps[None, :].T, metric=lambda x, y: np.abs(x-y) / np.max((x, y)))
    )
    mat = np.triu(mat)
    return mat


def compute_day_diffs(date_array):
    """Computes the forward diff in days given an array of datetime.date"""
    # Calculate forward date diffs
    ddiffs = []
    for i in range(1, len(date_array)):
        ddiffs.append(
            (date_array[i] - date_array[i-1]).days
        )
    ddiffs = np.array(ddiffs)
    return ddiffs


def diff_bin(array, thresh=.3):
    """ Bucketizes 1D arrays by splitting the values where their percent difference is greater than `thresh`.
    Returns array with "Cluster IDs" for the array
    """
    uniques = np.sort(np.abs(np.unique(array)))
    array_sorted = np.sort(uniques)
    # safe divide, 0 on NaN
    percent_diffs = np.divide(np.diff(array_sorted), array_sorted[1:], out=np.zeros_like(array_sorted[1:]), where=array_sorted[1:]!=0)
    markers = np.abs(percent_diffs) > thresh
    return np.digitize(np.abs(array), array_sorted[1:][markers])


def create_cluster_matrix(m_diff_days, centroids):
    """Computes clusters IDs (closest cluster ID) given a matrix and centroids using the L2 distance. """
    aux =  np.argsort(np.sqrt((m_diff_days[:,:][...,np.newaxis]-np.array(centroids))**2))[...,[0]]
    aux_reshape = aux.reshape(aux.shape[:2])
    return aux_reshape.astype(np.uint8)

def create_cluster_matrix_dbscan(matrix, eps=.3):
    """Computes clusters IDs of a matrix using DBSCAN"""
    clustmat = np.zeros_like(matrix)
    indices_upper_triangle = np.triu_indices(clustmat.shape[0])
    imps_clusters = DBSCAN(min_samples=3, eps=eps).fit_predict(matrix[indices_upper_triangle].flatten()[:, None])  # We only need to train on the upper triangle
    clustmat[indices_upper_triangle] = imps_clusters
    return clustmat.astype(np.int8)

def create_cluster_matrix_buckets(amounts, eps=.3):
    """Computes clusters IDs of a matrix using DBSCAN"""
    bin_ids = diff_bin(amounts, thresh=eps)
    clustmat = np.repeat(bin_ids[None,:], len(amounts), axis=0)
    return clustmat


def combine_matrices(diff_days_matrix, diff_amounts_matrix):
    left = np.array(diff_days_matrix).astype(str)
    right = np.array(diff_amounts_matrix).astype(str)

    left = np.core.defchararray.zfill(left, 3)  # PAd to three leadong 0s to ensure cluster ids are 
    
    mstr = np.char.add(left, 'D-')
    final_clust_mat = np.char.add(mstr, right)
    mstr = np.char.add(final_clust_mat, 'A')   # Matrix should have the form `<ID>D-<ID>A` for Days and Amounts repectively. This is done to prevent collisions
    return final_clust_mat


def filter_duplicate_paths(llists):
    """Takes a lists of paths and removes the (perfect) duplicate ones
    TODO: THIS IS A CONVENIENCE FUNC AND SHOULD NOT BE USED ON THE FINAL VERSION/PROD.
    ALSO BEWARE -> BETWEEN RUNS ORDER IN THE OUTPUT LIST CANT BE ASSURED.
    DO NOT USE ON PROD.
    """
    return list(map(list, list(set(frozenset(item) for item in llists))))


def main_matrix_method(datediffs,
                       amounts,
                       merge_matrices=True,
                       centroids=None,
                       use_dbscan=False,
                       thresh_amount=.3
                      ):
    """ Main matrix method for separating noise and signal from series of movements

    Parameters:
        - datediffs: Days differences between events
        - amounts: Movements amounts
        - centroids: Custom day centroids for computing the day difference matrix
        - use_dbscan: Whether to use DBSCAN or custom Bucket method for clustering amounts.
    """
    centroids = CENTROIDS if centroids is None else centroids

    m_diff_days = matrix_optim(datediffs, mask_zeros=True)
    m_diff_amnt= diff_imp_matrix_optim(amounts)

    # Cluster matrices (both days and amounts)
    m_clust_days = create_cluster_matrix(m_diff_days, CENTROIDS)
    if use_dbscan:
        amount_clust_mat = create_cluster_matrix_dbscan(m_diff_amnt, eps=thresh_amount)
    else:
        amount_clust_mat = create_cluster_matrix_buckets(amounts, eps=thresh_amount)

    mat = m_diff_days.astype(int)#.astype(str)


    if merge_matrices:
        mat = combine_matrices(m_clust_days, amount_clust_mat)

    llists = optim_paths(mat)
    llists = filter_duplicate_paths(llists)
    return llists




