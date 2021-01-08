import numpy as np
import time
from sklearn.base import clone, is_classifier
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed, Memory
from ._scoal import SCOAL

class MSCOAL(SCOAL):
    
    def __init__(self, 
                estimator=LinearRegression(), 
                tol = 1e-4, 
                max_iter = 100,
                minimize = True,
                test_size=0.2,
                init='random',
                random_state=42,
                n_jobs=1,
                cache=False,
                verbose=False):
        
        self.estimator = estimator
        self.tol = tol
        self.max_iter = max_iter
        self.test_size=test_size
        self.init = init
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cache=cache
        self.verbose = verbose
        self.is_classifier = is_classifier(estimator)

    def _split_row_clusters(self,data,fit_mask,test_mask,coclusters,models):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        results = self._compute_parallel(data,test_mask,coclusters,models,self._score_rows)
        scores = np.zeros((row_clusters.size,n_row_clusters))
        for i in range(n_row_clusters):
            for j in range(n_col_clusters):
                scores[:,i] += results[i*n_col_clusters+j] 
        cluster_to_split = scores.mean(axis=0).argmax()
        rows = np.where(row_clusters==cluster_to_split)[0]
        rows_scores = scores[row_clusters==cluster_to_split,cluster_to_split]
        rows = rows[np.argsort(rows)]
        rows_scores = np.sort(rows_scores)
        rows1 = np.array_split(rows[rows_scores==0],2)[1]
        rows2 = np.array_split(rows[rows_scores>0],2)[1]
        rows = np.concatenate((rows1,rows2))
        new_row_clusters = np.copy(row_clusters)
        new_row_clusters[rows] = n_row_clusters

        return new_row_clusters
    
    def _split_col_clusters(self,data,fit_mask,test_mask,coclusters,models):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        results = self._compute_parallel(data,test_mask,coclusters,models,self._score_cols)
        scores = np.zeros((col_clusters.size,n_col_clusters))
        for i in range(n_row_clusters):
            for j in range(n_col_clusters):
                scores[:,j] += results[i*n_col_clusters+j] 
        cluster_to_split = scores.mean(axis=0).argmax()
        cols = np.where(col_clusters==cluster_to_split)[0]
        cols_scores = scores[col_clusters==cluster_to_split,cluster_to_split]
        cols = cols[np.argsort(cols_scores)]
        cols_scores = np.sort(cols_scores)
        cols1 = np.array_split(cols[cols_scores==0],2)[1]
        cols2 = np.array_split(cols[cols_scores>0],2)[1]
        cols = np.concatenate((cols1,cols2))
        new_col_clusters = np.copy(col_clusters)
        new_col_clusters[cols] = n_col_clusters

        return new_col_clusters

    def _print_status(self,iter_count,score,delta_score,n_row_clusters,n_col_clusters,elapsed_time):
        if iter_count==0:
            print('|'.join(x.ljust(15) for x in [
                    'iteration',' score','delta score','n row clusters', 'n col clusters', 'elapsed time (s)']))

        print('|'.join(x.ljust(15) for x in ['%i' % iter_count,'%.4f' % score,'%.4f' % delta_score,'%i' % n_row_clusters,'%i'  % n_col_clusters,'%i' % elapsed_time]))

    def fit(self,matrix,row_features,col_features,fit_mask=None):
        np.random.seed(self.random_state) 
        data = matrix, row_features, col_features
        if fit_mask is None:
            fit_mask = np.invert(np.isnan(matrix)) 
        test = np.random.choice(np.arange(fit_mask.sum()), int(fit_mask.sum()*self.test_size),replace=False)
        rows,cols = np.where(fit_mask)
        test_mask = np.zeros(fit_mask.shape).astype(bool)
        test_mask[rows[test],cols[test]] = True
        fit_mask[rows[test],cols[test]] = False
        self.test_mask=test_mask
        self.fit_mask=fit_mask

        if self.cache:
            self.memory = Memory('./pyscoal-cache')
            self.method = self.memory.cache(self._cached_fit, ignore=['self','model','X','y','rows','cols'])

        iter_count=0 
        elapsed_time = 0
        score = np.nan
        delta_score=np.nan
        converged = False
        start = time.time()

        self.n_row_clusters,self.n_col_clusters = 1, 1
        self.coclusters = self._initialize_coclusters(fit_mask,self.n_row_clusters,self.n_col_clusters,how=self.init)
        self.models = self._initialize_models(fit_mask,self.coclusters)
        self.coclusters,self.models = self._converge_scoal(data,fit_mask,self.coclusters,self.models,False)
        score = np.sum(self._score_coclusters(data,test_mask,self.coclusters,self.models))/np.sum(test_mask)
        if self.verbose:
                self._print_status(iter_count,score,delta_score,self.n_row_clusters,self.n_col_clusters,elapsed_time)

        while not converged:

            row_clusters_changed = False
            col_clusters_changed = False

            coclusters, models = np.copy(self.coclusters),np.copy(self.models)
            rows_score = np.sum(self._score_coclusters(data,test_mask,coclusters,models))/np.sum(test_mask)
            new_row_clusters = self._split_row_clusters(data,fit_mask,test_mask,coclusters,models)
            coclusters[0] = new_row_clusters
            models = self._initialize_models(fit_mask,coclusters)
            coclusters,models = self._converge_scoal(data,fit_mask,coclusters,models,False)
            rows_delta_score = rows_score
            rows_score = np.sum(self._score_coclusters(data,test_mask,coclusters,models))/np.sum(test_mask)
            rows_delta_score -= rows_score
            if rows_delta_score>0:
                self.n_row_clusters+=1
                self.coclusters =  np.copy(coclusters)
                self.models = np.copy(models)
                row_clusters_changed = True

            coclusters, models = np.copy(self.coclusters),np.copy(self.models)
            cols_score = np.sum(self._score_coclusters(data,test_mask,coclusters,models))/np.sum(test_mask)
            new_col_clusters = self._split_col_clusters(data,fit_mask,test_mask,coclusters,models)
            coclusters[1] = new_col_clusters
            models = self._initialize_models(fit_mask,coclusters)
            coclusters,models = self._converge_scoal(data,fit_mask,coclusters,models,False)
            cols_delta_score = cols_score
            cols_score = np.sum(self._score_coclusters(data,test_mask,coclusters,models))/np.sum(test_mask)
            cols_delta_score -= cols_score
            if cols_delta_score>0:
                self.n_col_clusters+=1
                self.coclusters  = np.copy(coclusters)
                self.models = np.copy(models)
                col_clusters_changed = True

            delta_score = score
            score =  np.sum(self._score_coclusters(data,test_mask,self.coclusters,self.models))/np.sum(test_mask)
            delta_score -= score
            converged = not row_clusters_changed and not col_clusters_changed
            iter_count+=1
            elapsed_time = time.time() - start
            if self.verbose:
                self._print_status(iter_count,score,delta_score,self.n_row_clusters,self.n_col_clusters,elapsed_time)
        
        fit_mask = np.logical_or(fit_mask,test_mask)
        self.coclusters,self.models = self._converge_scoal(data,fit_mask,self.coclusters,self.models,False)
        self.elapsed_time = time.time() - start
        self.n_iter = iter_count
        if self.cache:
            self.memory.clear(warn=False)