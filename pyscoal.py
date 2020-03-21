import numpy as np
import time
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted

class SCOAL():
    
    def __init__(self, 
                estimator=LinearRegression(), 
                n_row_cluster = 2, 
                n_col_cluster = 2,
                tol = 1e-4, 
                max_iter = np.nan,
                scoring=mean_squared_error,
                minimize = True,
                init='random',
                inner_convergence=False,
                random_state=42,
                n_jobs=1,
                verbose=False):
        
        self.estimator = estimator
        self.n_row_cluster = n_row_cluster
        self.n_col_cluster = n_col_cluster
        self.tol = tol
        self.max_iter = max_iter
        self.scoring = scoring
        self.minimize = minimize
        self.init = init
        self.inner_convergence = inner_convergence
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _fit_model(self,matrix,row_features,col_features,fit_mask,row_cluster,col_cluster):

        row_cluster_mask = self.row_clusters==row_cluster
        col_cluster_mask = self.col_clusters==col_cluster
        cocluster_mask = np.logical_and(
            np.repeat(row_cluster_mask.reshape(-1,1),col_cluster_mask.shape,axis=1),
            np.repeat(col_cluster_mask.reshape(1,-1),row_cluster_mask.shape,axis=0)
        )

        idx = np.where(fit_mask & cocluster_mask)

        X = np.hstack((row_features[idx[0]],col_features[idx[1]]))
        y = matrix[idx].ravel()

        estimator = clone(self.estimator)
        estimator.fit(X,y)

        return estimator  

    def _fit_models(self,matrix,row_features,col_features,fit_mask): 

        estimators = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_model)(matrix,row_features,col_features,fit_mask,i,j) 
            for i in range(self.n_row_cluster) for j in range(self.n_col_cluster))

        estimators =  [[estimators[i*self.n_col_cluster+j] for j in range(self.n_col_cluster)] for i in range(self.n_row_cluster)]

        return estimators

    def _predict_model(self,matrix,row_features,col_features,fit_mask,row_cluster,col_cluster):

        prediction = np.copy(matrix)
        
        row_cluster_mask = self.row_clusters==row_cluster
        col_cluster_mask = self.col_clusters==col_cluster
        cocluster_mask = np.logical_and(
            np.repeat(row_cluster_mask.reshape(-1,1),col_cluster_mask.shape,axis=1),
            np.repeat(col_cluster_mask.reshape(1,-1),row_cluster_mask.shape,axis=0)
        )

        idx = np.where(fit_mask & cocluster_mask)

        X = np.hstack((row_features[idx[0]],col_features[idx[1]]))

        estimator = self.estimators[row_cluster][col_cluster]
        y_pred = estimator.predict(X)

        prediction[idx] = y_pred

        return prediction

    def _predict_models(self,matrix,row_features,col_features,fit_mask):

        predictions = Parallel(n_jobs=self.n_jobs)(delayed(self._predict_model)(matrix,row_features,col_features,fit_mask,i,j) 
            for i in range(self.n_row_cluster) for j in range(self.n_col_cluster))

        predictions = [[predictions[i*self.n_col_cluster+j] for j in range(self.n_col_cluster)] for i in range(self.n_row_cluster)]

        return predictions

    def _score_model(self,matrix,row_features,col_features,fit_mask,row_cluster,col_cluster):
        
        row_cluster_mask = self.row_clusters==row_cluster
        col_cluster_mask = self.col_clusters==col_cluster
        cocluster_mask = np.logical_and(
            np.repeat(row_cluster_mask.reshape(-1,1),col_cluster_mask.shape,axis=1),
            np.repeat(col_cluster_mask.reshape(1,-1),row_cluster_mask.shape,axis=0)
        )

        idx = np.where(fit_mask & cocluster_mask)

        X = np.hstack((row_features[idx[0]],col_features[idx[1]]))
        y = matrix[idx].ravel()

        estimator = self.estimators[row_cluster][col_cluster]
        y_pred = estimator.predict(X)

        score = self.scoring(y,y_pred)

        return score

    def _score_models(self,matrix,row_features,col_features,fit_mask):

        scores = Parallel(n_jobs=self.n_jobs)(delayed(self._score_model)(matrix,row_features,col_features,fit_mask,i,j) 
            for i in range(self.n_row_cluster) for j in range(self.n_col_cluster))

        scores = [[scores[i*self.n_col_cluster+j] for j in range(self.n_col_cluster)] for i in range(self.n_row_cluster)]

        return scores

    def _update_row_cluster(self,matrix,row_features,col_features,fit_mask,row):
        n_rows, n_cols = matrix.shape

        scores = np.zeros(self.n_row_cluster)

        row_mask = np.arange(n_rows)==row
        row_mask = np.repeat(row_mask.reshape(-1,1),n_cols,axis=1)

        for j in range(self.n_row_cluster):
            for k in range(self.n_col_cluster):

                row_cluster_mask = self.col_clusters==k
                col_cluster_mask = self.col_clusters==j
                cocluster_mask = np.logical_and(
                    np.repeat(row_cluster_mask.reshape(-1,1),col_cluster_mask.shape,axis=1),
                    np.repeat(col_cluster_mask.reshape(1,-1),row_cluster_mask.shape,axis=0)
                )
                
                idx = np.where(fit_mask & cocluster_mask & row_mask)

                X = np.hstack((row_features[idx[0]],col_features[idx[1]]))
                y = matrix[idx].ravel()

                y_pred = self.estimators[j][k].predict(X)

                scores[j] += self.scoring(y,y_pred)

            scores[j] = scores[j]/self.n_col_cluster
            
        new_row_cluster = np.argmin(scores) if self.minimize else np.argmax(scores)

        return new_row_cluster
    
    def _update_row_clusters(self,matrix,row_features,col_features,fit_mask):
        n_rows, _ = matrix.shape

        new_row_clusters = Parallel(n_jobs=self.n_jobs)(delayed(self._update_row_cluster)(matrix,row_features,col_features,fit_mask,i) 
            for i in range(n_rows))

        return new_row_clusters
        
    def _update_col_cluster(self,matrix,row_features,col_features,fit_mask,col):
        n_rows, n_cols = matrix.shape

        scores = np.zeros(self.n_col_cluster)

        col_mask = np.arange(n_rows)==col
        col_mask = np.repeat(col_mask.reshape(-1,1),n_cols,axis=1)

        for j in range(self.n_col_cluster):
            for k in range(self.n_row_cluster):

                row_cluster_mask = self.col_clusters==j
                col_cluster_mask = self.col_clusters==k
                cocluster_mask = np.logical_and(
                    np.repeat(row_cluster_mask.reshape(-1,1),col_cluster_mask.shape,axis=1),
                    np.repeat(col_cluster_mask.reshape(1,-1),row_cluster_mask.shape,axis=0)
                )
                
                idx = np.where(fit_mask & cocluster_mask & col_mask)

                X = np.hstack((row_features[idx[0]],col_features[idx[1]]))
                y = matrix[idx].ravel()

                y_pred = self.estimators[k][j].predict(X)

                scores[j] += self.scoring(y,y_pred)

            scores[j] = scores[j]/self.n_col_cluster
            
        new_col_cluster = np.argmin(scores) if self.minimize else np.argmax(scores)

        return new_col_cluster

    def _update_col_clusters(self,matrix,row_features,col_features,fit_mask):
        _, n_cols = matrix.shape
       
        new_col_clusters = Parallel(n_jobs=self.n_jobs)(delayed(self._update_row_cluster)(matrix,row_features,col_features,fit_mask,i) 
            for i in range(n_cols))

        return new_col_clusters

    def _check_cocluster(self,matrix,fit_mask,row_cluster,col_cluster):
        
        row_cluster_mask = self.row_clusters==row_cluster
        col_cluster_mask = self.col_clusters==col_cluster
        cocluster_mask = np.logical_and(
            np.repeat(row_cluster_mask.reshape(-1,1),col_cluster_mask.shape,axis=1),
            np.repeat(col_cluster_mask.reshape(1,-1),row_cluster_mask.shape,axis=0)
        )

        idx = np.where(fit_mask & cocluster_mask)

        y = matrix[idx].ravel()

        valid = y.size>0

        return  valid

    def check_coclusters(self,matrix,fit_mask):
        valids = Parallel(n_jobs=self.n_jobs)(delayed(self._check_cocluster)(matrix,fit_mask,i,j) 
            for i in range(self.n_row_cluster) for j in range(self.n_col_cluster))

        return valids

    def initialize_clustering(self,matrix):
        n_rows, n_cols = matrix.shape

        if self.init=='random':
            np.random.seed(self.random_state)  
            self.row_clusters = np.array([np.random.choice(np.arange(self.n_row_cluster)) for i in range(n_rows)])
            self.col_clusters = np.array([np.random.choice(np.arange(self.n_col_cluster)) for i in range(n_cols)])
        else: 
            self.row_clusters = np.array(self.init[0])
            self.col_clusters = np.array(self.init[1])
    
    def fit(self,matrix,row_features,col_features,fit_mask=None):
        
        iter_count=0 
        elapsed_time = 0
        rows_changed = 0
        cols_changed = 0
        score = np.nan
        delta_score=np.nan
        converged = False
        inner_converged= False

        if fit_mask is None:
            fit_mask = np.invert(np.isnan(matrix))
        

        self.estimators = [[clone(self.estimator) for i in range(self.n_col_cluster)] 
                            for j in range(self.n_row_cluster)]
        

        
        self.initialize_clustering(matrix)
        self._fit_models(matrix,row_features,col_features,fit_mask)
        score = np.mean(self._score_models(matrix,row_features,col_features,fit_mask))

        if self.verbose:
            print('|'.join(x.ljust(15) for x in [
                'iteration',' score','delta score','rows changed', 'columns changed', 'elapsed time (s)']))

            print('|'.join(x.ljust(15) for x in ['%i' % iter_count,
                                                    '%.3f' % score,
                                                    '%.3f' % delta_score,
                                                    '%i' % rows_changed,
                                                    '%i'  % cols_changed,
                                                    '%i' % elapsed_time]))
        start = time.time()
        while(not converged):
            rows_changed = 0
            cols_changed = 0
            inner_converged=False
            inner_count = 0 
            while (not inner_converged):

                new_row_clusters = self._update_row_clusters(matrix,row_features,col_features,fit_mask)
                inner_rows_changed = np.sum(new_row_clusters!=self.row_clusters)
                rows_changed += inner_rows_changed
                self.row_clusters = np.copy(new_row_clusters)

                new_col_clusters = self._update_col_clusters(matrix,row_features,col_features,fit_mask)
                inner_cols_changed = np.sum(new_col_clusters!=self.col_clusters)
                cols_changed += inner_cols_changed
                self.col_clusters = np.copy(new_col_clusters)
                
                inner_count+=1

                inner_converged = ( 
                                    not self.inner_convergence or
                                    (inner_rows_changed==0 and inner_cols_changed==0) or
                                    inner_count >= self.max_iter
                )

            delta_score = score
            self._fit_models(matrix,row_features,col_features,fit_mask)
            score = np.mean(self._score_models(matrix,row_features,col_features,fit_mask))
            delta_score -= score

            iter_count += 1

            converged = (
                iter_count >= self.max_iter or
                (delta_score > 0 and delta_score < self.tol) or
                (rows_changed==0 and cols_changed==0)
            )   

            elapsed_time = time.time() - start

            if self.verbose:
                print('|'.join(x.ljust(15) for x in ['%i' % iter_count,
                                                    '%.3f' % score,
                                                    '%.5f' % delta_score,
                                                    '%i' % rows_changed,
                                                    '%i'  % cols_changed,
                                                    '%i' % elapsed_time]))

    def predict(self,matrix,row_features,col_features,pred_mask=None):
        
        if pred_mask is None:
            pred_mask = np.isnan(matrix)

        prediction = self._predict_models(matrix,row_features,col_features,pred_mask)

        return prediction

    
    def score(self,matrix,row_features,col_features,score_mask=None):
        
        if score_mask is None:
            score_mask = np.invert(np.isnan(matrix))

        scores = np.mean(self._score_models(matrix,row_features,col_features,score_mask))
      
        return scores
