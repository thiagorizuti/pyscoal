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
                n_row_clusters = 2, 
                n_col_clusters = 2,
                tol = 1e-3, 
                max_iter = np.nan,
                scoring=mean_squared_error,
                minimize = True,
                init='random',
                inner_convergence=False,
                random_state=42,
                n_jobs=1,
                verbose=False):
        
        self.estimator = estimator
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.scoring = scoring
        self.minimize = minimize
        self.init = init
        self.inner_convergence = inner_convergence
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose


    def _get_mask(self,row_cluster,col_cluster):
        if row_cluster is None:
            rows_mask = np.ones(self.row_clusters.shape).astype(bool)
        else:
            rows_mask = self.row_clusters==row_cluster
        if col_cluster is None:
            cols_mask = np.ones(self.col_clusters.shape).astype(bool)
        else:
            cols_mask = self.col_clusters==col_cluster
        mask = np.logical_and(
            np.repeat(rows_mask.reshape(-1,1),cols_mask.shape,axis=1),
            np.repeat(cols_mask.reshape(1,-1),rows_mask.shape,axis=0)
        )

        return mask

    def _get_idx(self,row_cluster,col_cluster):
        if row_cluster is None:
            rows_mask = np.ones(self.row_clusters.shape).astype(bool)
        else:
            rows_mask = self.row_clusters==row_cluster
        if col_cluster is None:
            cols_mask = np.ones(self.col_clusters.shape).astype(bool)
        else:
            cols_mask = self.col_clusters==col_cluster
        idx = np.ix_(rows_mask,cols_mask)

        return idx

    def _get_X(self,row_features,col_features,mask=None,idx=None):
        if idx is not None:
            X = np.hstack([np.repeat(row_features[idx[0].ravel()], col_features[idx[1].ravel()].shape[0], axis=0),
                np.tile(col_features[idx[1].ravel()], (row_features[idx[0].ravel()].shape[0], 1))])
            X = X[self.fit_mask[idx].ravel()]
        if mask is not None: 
            rows, cols = np.where(mask&self.fit_mask)
            X = np.hstack((row_features[rows],col_features[cols]))

        return X

    def _get_y(self,matrix,mask=None,idx=None):
        if idx is not None:
            y = matrix[idx] 
            y = y[self.fit_mask[idx]]
        if mask is not None:
            rows, cols = np.where(mask&self.fit_mask)
            y = matrix[rows,cols].ravel()

        return y

    def _fit_model(self,X,y,estimator):
        estimator.fit(X,y)

        return estimator  

    def _predict_model(self,X,estimator):
        y_pred = estimator.predict(X)

        return y_pred

    def _score_model(self,X,y,estimator,scoring):
        y_pred = estimator.predict(X)
        score = scoring(y,y_pred)

        return score
       
    def _score_models_row_wise(self,matrix,row_features,col_features):

        predictions = Parallel(n_jobs=self.n_jobs)(delayed(self._predict_model)
            (self._get_X(row_features,col_features,idx=self._get_idx(None,j)),
                self.estimators[i][j])  
            for i in range(self.n_row_clusters) for j in range(self.n_col_clusters))

        scores = np.zeros((self.n_rows,self.n_row_clusters))
        for i in range(self.n_row_clusters):
            for j in range(self.n_col_clusters):
                true = matrix[self._get_idx(None,j)]
                pred = np.copy(true)
                pred[self.fit_mask[self._get_idx(None,j)]] = predictions[i*self.n_col_clusters+j]
                for r in range(self.n_rows):
                    y_true = true[r,:]
                    y_pred = pred[r,:]
                    scores[r,i] += self.scoring(y_true[~np.isnan(y_true)],y_pred[~np.isnan(y_pred)])
                    if np.isnan(y_true).all():
                        scores[r,i] += 0
                    else:
                        scores[r,i] += self.scoring(y_true[~np.isnan(y_true)],y_pred[~np.isnan(y_pred)])
            scores = scores/self.n_col_clusters
        return scores

    def _score_models_col_wise(self,matrix,row_features,col_features):

        predictions = Parallel(n_jobs=self.n_jobs)(delayed(self._predict_model)
            (self._get_X(row_features,col_features,idx=self._get_idx(i,None)),
                self.estimators[i][j])  
            for i in range(self.n_row_clusters) for j in range(self.n_col_clusters))

        scores = np.zeros((self.n_cols,self.n_col_clusters))
        for i in range(self.n_row_clusters):
            for j in range(self.n_col_clusters):
                true = matrix[self._get_idx(i,None)]
                pred = np.copy(true)
                pred[self.fit_mask[self._get_idx(i,None)]] = predictions[i*self.n_col_clusters+j]
                for c in range(self.n_cols):
                    y_true = true[:,c]
                    y_pred = pred[:,c]
                    if np.isnan(y_true).all():
                        scores[c,j] += 0
                    else:
                        scores[c,j] += self.scoring(y_true[~np.isnan(y_true)],y_pred[~np.isnan(y_pred)])
            scores = scores/self.n_row_clusters
        return scores
        
    # def _predict_models(self,matrix,row_features,col_features,fit_mask):
    #     predictions = Parallel(n_jobs=self.n_jobs)(delayed(self._predict_model)
    #         (self._get_X(row_features,col_features,i,j),self.estimators[i][j]) 
    #         for i in range(self.n_row_clusters) for j in range(self.n_col_clusters))

    #     predictions = [[predictions[i*self.n_col_clusters+j] 
    #         for j in range(self.n_col_clusters)] for i in range(self.n_row_clusters)]

    #     return predictions

    def _score_models(self,matrix,row_features,col_features):
        scores = Parallel(n_jobs=self.n_jobs)(delayed(self._score_model)
            (self._get_X(row_features,col_features,idx=self._get_idx(i,j)),
                self._get_y(matrix,idx=self._get_idx(i,j)),
                self.estimators[i][j], self.scoring) 
            for i in range(self.n_row_clusters) for j in range(self.n_col_clusters))

        scores = [[scores[i*self.n_col_clusters+j] 
            for j in range(self.n_col_clusters)] for i in range(self.n_row_clusters)]

        return scores

    def _fit_models(self,matrix,row_features,col_features): 
        estimators = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_model)
            (self._get_X(row_features,col_features,idx=self._get_idx(i,j)),
                self._get_y(matrix,idx=self._get_idx(i,j)),
                self.estimators[i][j]) 
            for i in range(self.n_row_clusters) for j in range(self.n_col_clusters))

        estimators =  [[estimators[i*self.n_col_clusters+j] 
            for j in range(self.n_col_clusters)] for i in range(self.n_row_clusters)]

        return estimators

    def _update_row_clusters(self,matrix,row_features,col_features):
        scores = self._score_models_row_wise(matrix,row_features,col_features)
        row_clusters  = np.argmin(scores,axis=1) if self.minimize else np.argmax(scores,axis=1)
        return row_clusters
    
    def _update_col_clusters(self,matrix,row_features,col_features):
        scores = self._score_models_col_wise(matrix,row_features,col_features)
        col_clusters  = np.argmin(scores,axis=1) if self.minimize else np.argmax(scores,axis=1)
        return col_clusters

    def _check_cocluster(self,row_cluster,col_cluster):
        valid = self.fit_mask[self._get_idx(row_cluster,col_cluster)].any()

        return  valid

    def check_coclusters(self):
        valids = Parallel(n_jobs=self.n_jobs)(delayed(self._check_cocluster)(i,j) 
            for i in range(self.n_row_clusters) for j in range(self.n_col_clusters))

        return valids

    def initialize_clustering(self):

        if self.init=='random':
            np.random.seed(self.random_state)  
            self.row_clusters = np.random.choice(np.arange(self.n_row_clusters),self.n_rows)
            self.col_clusters = np.random.choice(np.arange(self.n_col_clusters),self.n_cols)
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
            
        self.fit_mask = fit_mask
        self.estimators = [[clone(self.estimator) for i in range(self.n_col_clusters)] 
                            for j in range(self.n_row_clusters)]
        self.n_rows, self.n_cols = matrix.shape

        self.initialize_clustering()
        self.estimators=self._fit_models(matrix,row_features,col_features)
        score = np.mean(self._score_models(matrix,row_features,col_features))

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
                new_row_clusters = self._update_row_clusters(matrix,row_features,col_features)
                inner_rows_changed = np.sum(new_row_clusters!=self.row_clusters)
                rows_changed += inner_rows_changed
                self.row_clusters = np.copy(new_row_clusters)
                new_col_clusters = self._update_col_clusters(matrix,row_features,col_features)
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
            self.estimators=self._fit_models(matrix,row_features,col_features)
            score = np.mean(self._score_models(matrix,row_features,col_features))
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

    # def predict(self,matrix,row_features,col_features,pred_mask=None):
        
    #     if pred_mask is None:
    #         pred_mask = np.isnan(matrix)

    #     prediction = self._predict_models(matrix,row_features,col_features,pred_mask)

    #     return prediction

    
    # def score(self,matrix,row_features,col_features,score_mask=None):
        
    #     if score_mask is None:
    #         score_mask = np.invert(np.isnan(matrix))

    #     scores = np.mean(self._score_models(matrix,row_features,col_features,score_mask))
      
    #     return scores

    # def _predict_model(self,matrix,row_features,col_features,fit_mask,row_cluster,col_cluster):

    #         prediction = np.copy(matrix)
            
    #         row_cluster_mask = self.row_clusters==row_cluster
    #         col_cluster_mask = self.col_clusters==col_cluster
    #         cocluster_mask = np.logical_and(
    #             np.repeat(row_cluster_mask.reshape(-1,1),col_cluster_mask.shape,axis=1),
    #             np.repeat(col_cluster_mask.reshape(1,-1),row_cluster_mask.shape,axis=0)
    #         )

    #         idx = np.where(fit_mask & cocluster_mask)

    #         X = np.hstack((row_features[idx[0]],col_features[idx[1]]))

    #         estimator = self.estimators[row_cluster][col_cluster]
    #         y_pred = estimator.predict(X)

    #         prediction[idx] = y_pred

    #         return prediction