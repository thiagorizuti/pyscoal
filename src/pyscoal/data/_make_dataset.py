import numpy as np

def _make_coefs(n_row_clusters,n_col_clusters,n_row_features,n_col_features):
    n_models = n_row_clusters*n_col_clusters
    n_features = n_row_features+n_col_features
    coefs = np.random.uniform(-1*n_models,n_models,(n_row_clusters,n_col_clusters,n_features))
    return coefs

def _make_features(n_rows,n_cols,n_row_features,n_col_features):
    row_features = np.random.uniform(0,1,(n_rows,n_row_features))
    col_features = np.random.uniform(0,1,(n_cols,n_col_features))
    return row_features, col_features

def _make_coclusters(n_rows,n_cols,n_row_clusters,n_col_clusters):
    row_clusters = np.random.choice(np.arange(n_row_clusters),n_rows)
    col_clusters = np.random.choice(np.arange(n_col_clusters),n_cols)        
    return row_clusters,col_clusters


def _make_target(coefs,row_features,col_features,row_clusters,col_clusters):
    n_rows, n_cols = row_features.shape[0], col_features.shape[0]
    target = np.hstack([np.repeat(np.arange(n_rows), n_cols, axis=0).reshape(-1,1), 
        np.tile(np.arange(n_cols), n_rows).reshape(-1,1),np.zeros(n_rows*n_cols).reshape(-1,1)])
    for row_cluster in np.unique(row_clusters):
            for col_cluster in np.unique(col_clusters):
                rows = row_clusters[target[:,0].astype(int)]==row_cluster
                cols = col_clusters[target[:,1].astype(int)]==col_cluster

                features = np.hstack((row_features[target[rows&cols,0].astype(int),:],
                    col_features[target[rows&cols,1].astype(int),:]))

                target[rows&cols,-1] = np.dot(features,coefs[row_cluster,col_cluster,:].T)

    return target

def make_dataset(n_rows=100,n_cols=100,
                 n_row_clusters=2,n_col_clusters=2,
                 n_row_features=5,n_col_features=5,
                 noise=0,sparsity=0,
                 seed=42):
    np.random.seed(seed)
    
    coefs = _make_coefs(n_row_clusters,n_col_clusters,n_row_features,n_col_features)
    row_features, col_features = _make_features(n_rows,n_cols,n_row_features,n_col_features)
    row_clusters, col_clusters = _make_coclusters(n_rows,n_cols,n_row_clusters,n_col_clusters)
    target = _make_target(coefs,row_features,col_features,row_clusters,col_clusters)
    
    if sparsity > 0:
        keep = np.full(target.shape[0],False)
        keep[:int(target.shape[0]*(1-sparsity))] = True
        np.random.shuffle(keep)
        target = target[keep,:]
    
    target[:,-1] += np.random.normal(0,noise,(target.shape[0]))
    
    return target, row_features, col_features, row_clusters, col_clusters, coefs
