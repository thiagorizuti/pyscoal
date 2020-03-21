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
                metric=mean_squared_error,
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
        self.metric = metric
        self.init=init
        self.inner_convergence=inner_convergence
        self.random_state=random_state
        self.verbose=verbose
        self.n_jobs=n_jobs
        self.error=np.nan

    
    def initialize_clustering(self,matrix,row_features,col_features,fit_mask):

        n_rows, n_cols = matrix.shape
        self.estimators = [[clone(self.estimator) for i in range(self.n_col_cluster)] 
                            for j in range(self.n_row_cluster)]
        if self.init=='uniform':
            self.row_clusters = np.zeros(n_rows)
            splits = np.array_split(np.arange(n_rows),self.n_row_cluster)
            for i,s in enumerate(splits):
                self.row_clusters[s] = i
            self.col_clusters = np.zeros(n_cols)
            splits = np.array_split(np.arange(n_cols),self.n_col_cluster)
            for i,s in enumerate(splits):
                self.col_clusters[s] = i

        elif self.init=='random':
            np.random.seed(self.random_state)  
            self.row_clusters = np.array([np.random.choice(np.arange(self.n_row_cluster)) for i in range(n_rows)])
            self.col_clusters = np.array([np.random.choice(np.arange(self.n_col_cluster)) for i in range(n_cols)])

        else: 
            self.row_clusters = np.array(self.init[0])
            self.col_clusters = np.array(self.init[1])

        if fit_mask is None:
            fit_mask = np.invert(np.isnan(matrix))
        error = self._fit_models(matrix,row_features,col_features,fit_mask)

        return error

    def _fit_model(self,matrix,row_features,col_features,mask,i,j):

        row_cluster_mask = self.row_clusters==i
        col_cluster_mask = self.col_clusters==j
        cocluster_mask = np.logical_and(
            np.repeat(row_cluster_mask.reshape(-1,1),col_cluster_mask.shape,axis=1),
            np.repeat(col_cluster_mask.reshape(1,-1),row_cluster_mask.shape,axis=0)
        )

        idx = np.where(mask & cocluster_mask)

        X = np.hstack((row_features[idx[0]],col_features[idx[1]]))
        y = matrix[idx].ravel()

        estimator = clone(self.estimators[i][j])
        estimator.fit(X,y)
        y_pred = estimator.predict(X)
        error = self.metric(y,y_pred)

        return error, estimator  

    def _fit_models(self,matrix,row_features,col_features,mask):        
        results = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_model)(matrix,row_features,col_features,mask,i,j) 
            for i in range(self.n_row_cluster) for j in range(self.n_col_cluster))

        errors = [r[0] for r in results]
        estimators = [r[1] for r in results]
        for i in range(self.n_row_cluster):
            for j in range(self.n_col_cluster):
                self.estimators[i][j] = estimators[i*j]
        return np.mean(errors) 
    
        
    def _update_row_cluster(self,matrix,row_features,col_features,fit_mask):
        n_rows, _ = matrix.shape
        new_row_clusters = np.copy(self.row_clusters)
        for i in range(n_rows):
            error = np.zeros(self.n_row_cluster)
            for j in range(self.n_row_cluster):
                for k in range(self.n_col_cluster):
                        sub_matrix = matrix[i,self.col_clusters==k]
                        sub_row_features = row_features[i,:][None,:]
                        sub_col_features = col_features[self.col_clusters==k,:]

                        X = np.concatenate([np.repeat(sub_row_features,sub_col_features.shape[0],axis=0),
                        sub_col_features],axis=1)
                        y = sub_matrix.ravel()
                        
                        mask = fit_mask[i,self.col_clusters==k].ravel()
                        y = y[mask]
                        X = X[mask]
                        if y.size > 0:
                            y_pred = self.estimators[j][k].predict(X)
                            error[j] += self.metric(y,y_pred)
                error[j] = error[j]/self.n_col_cluster
            new_row_clusters[i] = np.argmin(error)
        return new_row_clusters
        
    def _update_col_cluster(self,matrix,row_features,col_features,fit_mask):
        _, n_cols = matrix.shape
        new_col_clusters = np.copy(self.col_clusters)
        for i in range(n_cols):
            error = np.zeros(self.n_col_cluster)
            for j in range(self.n_col_cluster):
                for k in range(self.n_row_cluster): 
                        sub_matrix = matrix[self.row_clusters==k,i]
                        sub_row_features = row_features[self.row_clusters==k,:]
                        sub_col_features = col_features[i,:][None,:]
                        
                        X = np.concatenate([sub_row_features,
                        np.repeat(sub_col_features,sub_row_features.shape[0],axis=0)],axis=1)
                        y = sub_matrix.ravel()

                        mask = fit_mask[self.row_clusters==k,i].ravel()
                        y = y[mask]
                        X = X[mask]
                        
                        if y.size > 0:
                            y_pred = self.estimators[k][j].predict(X)
                            error[j] += self.metric(y,y_pred)
                error[j] = error[j]/self.n_row_cluster
            new_col_clusters[i] = np.argmin(error)
        return new_col_clusters
        
    def fit(self,matrix,row_features,col_features,mask=None):
        
        iter_count=0 
        elapsed_time = 0
        rows_changed = 0
        cols_changed = 0
        delta_error=np.nan
        converged = False
        inner_converged= False

        fit_mask = np.invert(np.isnan(matrix))
        
        self.error = self.initialize_clustering(matrix,row_features,col_features,fit_mask)
        #delta_error = self.__fit_models(matrix,row_features,col_features,fit_mask)

        if self.verbose:
            print('|'.join(x.ljust(15) for x in [
                'iteration',' error','delta error','rows changed', 'columns changed', 'elapsed time (s)']))

            print('|'.join(x.ljust(15) for x in ['%i' % iter_count,
                                                    '%.3f' % self.error,
                                                    '%.3f' % delta_error,
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

                new_row_clusters = self._update_row_cluster(matrix,row_features,col_features,fit_mask)
                inner_rows_changed = np.sum(new_row_clusters!=self.row_clusters)
                rows_changed += inner_rows_changed
                self.row_clusters = np.copy(new_row_clusters)

                new_col_clusters = self._update_col_cluster(matrix,row_features,col_features,fit_mask)
                inner_cols_changed = np.sum(new_col_clusters!=self.col_clusters)
                cols_changed += inner_cols_changed
                self.col_clusters = np.copy(new_col_clusters)
                
                inner_count+=1

                inner_converged = ( 
                                    not self.inner_convergence or
                                    (inner_rows_changed==0 and inner_cols_changed==0) or
                                    inner_count >= 4
                )

            delta_error = self.error
            self.error = self._fit_models(matrix,row_features,col_features,fit_mask)
            delta_error -= self.error

            iter_count += 1

            converged = (
                iter_count >= self.max_iter or
                (delta_error > 0 and delta_error < self.tol) or
                (rows_changed==0 and cols_changed==0)
            )   

            elapsed_time = time.time() - start

            if self.verbose:
                print('|'.join(x.ljust(15) for x in ['%i' % iter_count,
                                                    '%.3f' % self.error,
                                                    '%.5f' % delta_error,
                                                    '%i' % rows_changed,
                                                    '%i'  % cols_changed,
                                                    '%i' % elapsed_time]))

    def predict(self,matrix,row_features,col_features,mask=None):
        
        if mask is None:
            mask = np.isnan(matrix)

        prediction = np.copy(matrix)

        for i in range(self.n_row_cluster):
            for j in range(self.n_col_cluster):

                row_cluster_mask = self.row_clusters==i
                col_cluster_mask = self.col_clusters==j
                cocluster_mask = np.logical_and(
                    np.repeat(row_cluster_mask.reshape(-1,1),col_cluster_mask.shape,axis=1),
                    np.repeat(col_cluster_mask.reshape(1,-1),row_cluster_mask.shape,axis=0)
                )

                idx = np.where(mask & cocluster_mask)

                X = np.hstack((row_features[idx[0]],col_features[idx[1]]))

                y_pred = self.estimators[i][j].predict(X)

                prediction[idx] = y_pred
        return prediction

    
    def score(self,matrix,row_features,col_features,mask=None):
        score = 0 

        if mask is None:
            mask = np.invert(np.isnan(matrix))

        for i in range(self.n_row_cluster):
            for j in range(self.n_col_cluster):

                row_cluster_mask = self.row_clusters==i
                col_cluster_mask = self.col_clusters==j
                cocluster_mask = np.logical_and(
                    np.repeat(row_cluster_mask.reshape(-1,1),col_cluster_mask.shape,axis=1),
                    np.repeat(col_cluster_mask.reshape(1,-1),row_cluster_mask.shape,axis=0)
                )

                idx = np.where(mask & cocluster_mask)

                X = np.hstack((row_features[idx[0]],col_features[idx[1]]))
                y = matrix[idx].ravel()

                y_pred = self.estimators[i][j].predict(X)

                score += self.metric(y,y_pred)

        return score/(self.n_row_cluster*self.n_col_cluster)
