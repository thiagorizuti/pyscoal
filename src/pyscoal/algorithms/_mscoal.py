import numpy as np
import time
from sklearn.base import is_regressor
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed, Memory
from copy import deepcopy
from ._scoal import SCOAL

class MSCOAL(SCOAL):
    
    def __init__(self, 
                estimator=LinearRegression(), 
                max_split = 10,
                validation_size=0.2,
                random_state=42,
                n_jobs=1,
                cache=False,
                matrix='sparse',
                verbose=False):
        
        self.estimator = estimator
        self.max_split = max_split
        self.validation_size=validation_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cache=cache
        self.matrix=matrix
        self.verbose = verbose
        self.is_regressor = is_regressor(estimator)
    
    def _split_row_clusters(self,data,coclusters,models,n_jobs):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        results = self._compute_clusterwise(data,coclusters,models,self._score_rows,n_jobs)
        scores = np.zeros((row_clusters.size,n_row_clusters))
        for i in range(n_row_clusters):
            for j in range(n_col_clusters):
                scores[:,i] += results[i][j] 
        cluster_to_split = scores.mean(axis=0).argmax()
        rows = np.where(row_clusters==cluster_to_split)[0]
        rows_scores = scores[row_clusters==cluster_to_split,cluster_to_split]
        rows = rows[np.argsort(rows)]
        rows_scores = np.sort(rows_scores)
        rows1 = np.array_split(rows[rows_scores==0],2)[1]
        rows2 = np.array_split(rows[rows_scores>0],2)[1]
        rows = np.concatenate((rows1,rows2))
        new_row_clusters = row_clusters
        new_row_clusters[rows] = n_row_clusters

        return new_row_clusters, col_clusters
    
    def _split_col_clusters(self,data,coclusters,models,n_jobs):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        results = self._compute_clusterwise(data,coclusters,models,self._score_cols,n_jobs)
        scores = np.zeros((col_clusters.size,n_col_clusters))
        for i in range(n_row_clusters):
            for j in range(n_col_clusters):
                scores[:,j] += results[i][j] 
        cluster_to_split = scores.mean(axis=0).argmax()
        cols = np.where(col_clusters==cluster_to_split)[0]
        cols_scores = scores[col_clusters==cluster_to_split,cluster_to_split]
        cols = cols[np.argsort(cols_scores)]
        cols_scores = np.sort(cols_scores)
        cols1 = np.array_split(cols[cols_scores==0],2)[1]
        cols2 = np.array_split(cols[cols_scores>0],2)[1]
        cols = np.concatenate((cols1,cols2))
        new_col_clusters = col_clusters
        new_col_clusters[cols] = n_col_clusters

        return row_clusters, new_col_clusters

    def _print_status(self,iter_count,score,delta_score,n_row_clusters,n_col_clusters,elapsed_time):
        if iter_count==0:
            print('|'.join(x.ljust(15) for x in [
                    'split',' score','delta score','n row clusters', 'n col clusters', 'elapsed time (s)']))

        print('|'.join(x.ljust(15) for x in ['%i' % iter_count,'%.4f' % score,'%.4f' % delta_score,'%i' % n_row_clusters,'%i'  % n_col_clusters,'%i' % elapsed_time]))


    def _converge_mscoal(self,train_data,valid_data,coclusters,models,max_split=100,n_jobs=1,verbose=False):
        split_count=0 
        elapsed_time = 0
        score = np.nan
        delta_score=np.nan
        converged = False
        start = time.time()

        if coclusters is None:
            n_row_clusters, n_col_clusters = 1, 1
            coclusters = self._initialize_coclusters(n_row_clusters,n_col_clusters)
        else:
            row_clusters, col_clusters = coclusters
            n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size

        if models is None:
            models = self._initialize_models(coclusters)

        coclusters,models = self._converge_scoal(train_data,coclusters,models,n_jobs=n_jobs,verbose=False)
        scores = self._score_coclusters(valid_data,coclusters,models,n_jobs)
        score = np.sum(scores)/int(self.n_values*self.validation_size)
        
        if verbose:
                self._print_status(split_count,score,delta_score,n_row_clusters,n_col_clusters,elapsed_time)

        converged = split_count >= max_split

        while not converged:

            row_clusters_changed = False
            col_clusters_changed = False
            delta_score = 0

            new_coclusters = deepcopy(coclusters)
            new_coclusters = self._split_row_clusters(valid_data,new_coclusters,models,n_jobs)
            new_models = self._initialize_models(new_coclusters)
            checked = np.all(self._check_coclusters(train_data,new_coclusters,models,n_jobs=1))
            if checked:
                new_coclusters,new_models = self._converge_scoal(train_data,new_coclusters,new_models,n_jobs=n_jobs,verbose=False)
                scores = self._score_coclusters(valid_data,new_coclusters,new_models,n_jobs)
                new_score = np.sum(scores)/int(self.n_values*self.validation_size)
                new_delta_score = score - new_score
                if new_delta_score>0:
                    n_row_clusters+=1
                    coclusters =  new_coclusters
                    models = new_models
                    row_clusters_changed = True
                    delta_score += new_delta_score
                    score = new_score

            new_coclusters = deepcopy(coclusters)
            new_coclusters = self._split_col_clusters(valid_data,new_coclusters,models,n_jobs)
            new_models = self._initialize_models(new_coclusters)
            checked = np.all(self._check_coclusters(train_data,new_coclusters,models,n_jobs=1))
            if checked:
                new_coclusters,new_models = self._converge_scoal(train_data,new_coclusters,new_models,n_jobs=n_jobs,verbose=False)
                scores = self._score_coclusters(valid_data,new_coclusters,new_models,n_jobs)
                new_score = np.sum(scores)/int(self.n_values*self.validation_size)
                new_delta_score = score - new_score
                if new_delta_score>0:
                    n_col_clusters+=1
                    coclusters  = new_coclusters
                    models = new_models
                    col_clusters_changed = True
                    delta_score += new_delta_score
                    score = new_score

            converged = (not row_clusters_changed and not col_clusters_changed) or split_count >= max_split
            split_count+=1
            elapsed_time = time.time() - start
            if verbose:
                self._print_status(split_count,score,delta_score,n_row_clusters,n_col_clusters,elapsed_time)
        
        train_matrix, row_features, col_features = train_data
        valid_matrix, _, _ = valid_data
        if self.matrix=='dense':
            train_matrix[np.where(np.invert(np.isnan(valid_matrix)))] = valid_matrix[np.where(np.invert(np.isnan(valid_matrix)))]
        else:
            train_matrix = np.vstack((train_matrix,valid_matrix))
        train_data = (train_matrix,row_features,col_features)

        coclusters, models = self._converge_scoal(train_data,coclusters,models,n_jobs=n_jobs,verbose=False)

        return coclusters,models

    
    def fit(self,target,row_features,col_features,coclusters=None):
        np.random.seed(self.random_state) 
        
        self.n_rows, self.n_cols, self.n_values = row_features.shape[0], col_features.shape[0], target.shape[0]
        self.n_row_features, self.n_col_features  = row_features.shape[1], col_features.shape[1]        
        
        valid = np.full(self.n_values,False)
        valid[:int(self.n_values*self.validation_size)] = True
        np.random.shuffle(valid)
        valid_target = target[valid]
        train_target = target[~valid] 
        del target

        if self.matrix=='dense':
            valid_matrix = np.zeros((self.n_rows, self.n_cols))*np.nan
            valid_matrix[valid_target[:,0].astype(int),valid_target[:,1].astype(int)] = valid_target[:,2]
  
            train_matrix = np.zeros((self.n_rows, self.n_cols))*np.nan
            train_matrix[train_target[:,0].astype(int),train_target[:,1].astype(int)] = train_target[:,2]      
        else:
            valid_matrix = valid_target  
            train_matrix = train_target 
        del train_target
        del valid_target     
        
        valid_data = (valid_matrix,row_features,col_features)
        train_data = (train_matrix,row_features,col_features)

        if self.cache:
            self.memory = Memory('./pyscoal-cache')
            self._cached_fit = self.memory.cache(self._cached_fit, ignore=['self','model','X','y'])\
        
        self.coclusters,self.models = self._converge_mscoal(train_data,valid_data,coclusters,None,self.max_split,self.n_jobs,self.verbose)
        row_clusters, col_clusters = self.coclusters
        self.n_row_clusters, self.n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size

        if self.cache:
            self.memory.clear(warn=False)