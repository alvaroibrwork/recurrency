import numpy as np

from scipy import spatial
from sklearn.cluster import DBSCAN
import networkx as nx
from tqdm import tqdm


centroids = [2.60999494,16.62854731,30.21092294,44.44653696,61.54627043,91.75547689,134.94938152,195.94670594, 336.74164796,527.55770016,775.6417704, 100000]
CENTROIDS = centroids

def matrix_optim(difsarr, mask_zeros=True):
    """Computes the pairwise distances of days, given the array of sequencial differences (as ints)"""
    d = np.concatenate(([0], difsarr.cumsum()))[None, :]  # original Vero's behavior with duplicated first entry
    # d = difsarr.cumsum()[None, :]   # Rework
    dists = spatial.distance.pdist(d.T,
                                   metric='cityblock')  # TODO: sklearn has the pairwise distance, which supports parallelization (idk if there is any speedup)
    mat = spatial.distance.squareform(dists)
    mat = np.triu(mat)
    # mat = np.tril(mat)

    if mask_zeros:
        mat[np.where(mat == 0)] = 10000000

    return mat

def diff_bin(array, thresh=.3):
    """ 
    Bucketizes 1D arrays by splitting the values where their percent difference is greater than `thresh`.

    Returns array with "Cluster IDs" for the array
    """
    uniques = np.sort(np.abs(np.unique(array))).astype(float)
    array_sorted = np.sort(uniques)
    # safe divide, 0 on NaN
    percent_diffs = np.divide(np.diff(array_sorted), array_sorted[1:], out=np.zeros_like(array_sorted[1:]), where=array_sorted[1:]!=0)
    markers = np.abs(percent_diffs) > thresh
    return np.digitize(np.abs(array), array_sorted[1:][markers])

def create_cluster_matrix(m_diff_days, centroids=None):
    if centroids is None:
        centroids = CENTROIDS
        
    aux =  np.argsort(np.sqrt((m_diff_days[:,:][...,np.newaxis]-np.array(centroids))**2))[...,[0]]
    aux_reshape = aux.reshape(aux.shape[:2])
    return aux_reshape


def diff_amount_matrix(time_series):
    """
    from all elements of the amounts array (time_series), substract  the first one, 
    then the second one and so on.. 
      
        a0-a0, a1-a0, a2-a0, a3-a0, a4-a0,
        
        a0-a1, a1-a1, a2-a1, ...
        
    then we reshape this differences to return an upper triangular matrix where element (i,j) 
    is the absolute distance of j and i.
        
    """
    a = np.array(time_series)
    pairWiseDiff = np.array([])
    for i in a:
        individualDiff = (a - i)/float(i)
        pairWiseDiff = np.append(pairWiseDiff, individualDiff)
       
    # create a matrix from the 1 dimensional array 
    diff_matrix=np.reshape(pairWiseDiff, (len(a), len(a)))
    # keep only upper triangle
    diff_matrix_triu_abs=np.abs(np.triu(diff_matrix, 0))
    #output_matrix=diff_imp_matrix_triu_abs.tolist()
    
    return np.array(diff_matrix_triu_abs)

def define_matrix_combined_dbscan(m_clusters, amounts, min_samples = 3, eps = 0.3):
    
    m = diff_amount_matrix(amounts)
    a_clusters = DBSCAN(min_samples = min_samples, eps = eps).fit_predict(m.flatten()[:, None])
    a_clusters[a_clusters < 0 ] = 99 
    a_clusters = a_clusters.reshape(m.shape).astype(str)

    #combine
    mstr = np.array(m_clusters).astype(str)
    #mstr = np.char.add(mstr, '-')
    final_clust_mat = np.char.add(mstr, a_clusters)

    return (final_clust_mat.astype(float))

def define_matrix_combined(m_clusters, amounts_array, thresh=0.3):

    #print('------ binning amounts result -------')
    amt_bin = diff_bin(array = amounts_array, thresh = thresh )
    #print(amt_bin)
    
    #M =  np.zeros((m_clusters.shape[0],m_clusters.shape[0])) 
    M = np.full((m_clusters.shape[0],m_clusters.shape[0]), 999)
    #print(M)
    for i in range(0,m_clusters.shape[0]):
    
        for j in range(i+1,m_clusters.shape[0]):
            if amt_bin[i] == amt_bin[j]:
                
                M[i,j] = amt_bin[i] 
            else:
                M[i,j] = i*100+j
 
    #combine
    mstr = np.array(m_clusters).astype(str)
    #mstr = np.char.add(mstr, '-')
    final_clust_mat = np.char.add(mstr, M.astype(str))
    
    return (final_clust_mat.astype(float))






#P_BAR = tqdm(range(30))

#used only a matrix as input, not the node binning
def graph_matrix_original(matrix_numpy):
    
    """
    Given a numpy weighted adjacency matrix, create a directed weighted graph.
    
    We traverse the subgraphs corresponding to different edge-weights to find 
    all disjoint longest paths, iteratively. Then we return the longest overall disjoint paths.
   
    """
    
    A = np.array(matrix_numpy)
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    uw = np.unique(matrix_numpy) 

    paths_final = []

    for cluster in uw:

        #one subgraph for each cluster and longest path for each subgraph
        SG = nx.DiGraph( [ (u,v,d) for u,v,d in G.edges(data=True) if (d['weight']==cluster and u<v)]) 

        longest_paths = nx.dag_longest_path(SG)
        if len(longest_paths)>=3:
            paths_final.append(longest_paths)

        
        while len(longest_paths)>=3:
            #P_BAR.update(1)
            
            SG.remove_nodes_from(longest_paths)
            #longest paths after removing vertices from first longest path
            # TODO: try to improve by avoiding the computation of this function every time.
            # By running it once, we could cache the results.
            # Cache the topological sorting Networkx does.
            longest_paths = nx.dag_longest_path(SG)
            if len(longest_paths)>=3:
                paths_final.append(longest_paths)
        

    paths_final_disjoint = disjoint_lists (paths_final)
    #P_BAR.reset()

    return paths_final_disjoint

def disjoint_lists (input_l):
    
    #order from longest to shortest list
    input_l.sort(key=len, reverse=True)
    
    candidate = input_l.copy()

    for i in range(0, len(input_l)-1):

        input_l = candidate.copy()
        
        if i > len(input_l):
            break
        #reset the input list to only compute intersection between lists that remain as candidates
 
        for j in range(i+1,len(candidate)):
            
            #if two list have non null interception, remove the second list, which is shorter
            if set(input_l[i]) & set(input_l[j]):
                candidate.remove(input_l[j])
                
    return candidate


def main_matrix_method_graphs(datediffs,
                       amounts,
                       merge_matrices=True,
                       centroids=None,
                       use_dbscan=False,
                       ):
    
    centroids_ori = [2.60999494, 16.62854731, 30.21092294, 44.44653696, 61.54627043, 91.75547689, 134.94938152,
                     195.94670594, 336.74164796, 527.55770016, 775.6417704, 100000]

    centroids = centroids_ori if centroids is None else centroids

    m_diff_days = matrix_optim(datediffs)

    m_clusters = create_cluster_matrix(m_diff_days,centroids)
    
    #the way in which we build the amounts matrix changes 
    if use_dbscan:
        combined_clust_mat = define_matrix_combined_dbscan(m_clusters, amounts, min_samples = 2, eps = 0.3)
    else:
        combined_clust_mat = define_matrix_combined(m_clusters, amounts, thresh=0.3)
    
    paths = []
    paths = graph_matrix_original(combined_clust_mat)   
    
    return paths