import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class SCOAL():
    
    def __init__(self, estimator=LinearRegression(), 
                 n_row_cluster = 2, 
                 n_col_cluster = 2, 
                 tol = 0.0001, 
                 max_iter = np.nan,
                 early_stop = 0,
                 metric=mean_squared_error,
                 row_clusters=None,
                 col_clusters=None):
        
        self.estimator = estimator
        self.n_row_cluster = n_row_cluster
        self.n_col_cluster = n_col_cluster
        self.tol = tol
        self.metric = metric
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.row_clusters=None
        self.col_clusters=None
        self.error=np.nan
    
    def __initialize_clustering(self,matrix):
        n_rows, n_cols = matrix.shape
        self.estimators = [[clone(self.estimator) for n in range(self.n_row_cluster)] 
                           for n in range(self.n_col_cluster)]
        self.row_clusters = np.array([np.random.choice(np.arange(self.n_row_cluster)) for i in range(n_rows)])
        self.col_clusters = np.array([np.random.choice(np.arange(self.n_col_cluster)) for i in range(n_cols)])
   
    def __fit_models(self,matrix,row_features,col_features,fit_mask):
        for i in range(self.n_row_cluster):
            for j in range(self.n_col_cluster):
                sub_matrix = matrix[np.ix_(self.row_clusters==i,self.col_clusters==j)]
                sub_row_features = row_features[self.row_clusters==i,:]
                sub_col_features = col_features[self.col_clusters==j,:]
 
                X = np.hstack([np.repeat(sub_row_features, sub_col_features.shape[0], axis=0),
                    np.tile(sub_col_features, (sub_row_features.shape[0], 1))])
                y = sub_matrix.ravel()
                mask = fit_mask[np.ix_(self.row_clusters==i,self.col_clusters==j)].ravel()
                y = y[mask]
                X = X[mask]

                self.estimators[i][j].fit(X,y)

    def __compute_error(self,matrix,row_features,col_features,fit_mask):
        error = 0
        for i in range(self.n_row_cluster):
            for j in range(self.n_col_cluster):
                sub_matrix = matrix[np.ix_(self.row_clusters==i,self.col_clusters==j)]
                sub_row_features = row_features[self.row_clusters==i,:]
                sub_col_features = col_features[self.col_clusters==j,:]
 
                X = np.hstack([np.repeat(sub_row_features, sub_col_features.shape[0], axis=0),
                    np.tile(sub_col_features, (sub_row_features.shape[0], 1))])
                y = sub_matrix.ravel()

                mask = fit_mask[np.ix_(self.row_clusters==i,self.col_clusters==j)].ravel()
                y = y[mask]
                X = X[mask]

                y_pred = self.estimators[i][j].predict(X)
                
                error += self.metric(y,y_pred)
        self.error=error

    def __update_row_cluster(self,matrix,row_features,col_features,fit_mask):
        n_rows, _ = matrix.shape
        new_row_clusters = np.copy(self.row_clusters)
        for i in range(n_rows):
            error = np.zeros(self.n_row_cluster)
            for j in range(self.n_col_cluster):
                    sub_matrix = matrix[i,self.col_clusters==j]
                    sub_row_features = row_features[i,:][None,:]
                    sub_col_features = col_features[self.col_clusters==j,:]
                    X = np.concatenate([np.repeat(sub_row_features,sub_col_features.shape[0],axis=0),sub_col_features],axis=1)
                    y = sub_matrix.ravel()
                    
                    mask = fit_mask[i,self.col_clusters==j].ravel()
                    y = y[mask]
                    X = X[mask]

                    for k in range(self.n_row_cluster):
                        y_pred = self.estimators[k][j].predict(X)
                        error[k] += self.metric(y,y_pred)
            new_row_clusters[i] = np.argmin(error)
        return new_row_clusters
        
    def __update_col_cluster(self,matrix,row_features,col_features,fit_mask):
        _, n_cols = matrix.shape
        new_col_clusters = np.copy(self.col_clusters)
        for i in range(n_cols):
            error = np.zeros(self.n_col_cluster)
            for j in range(self.n_row_cluster):
                sub_matrix = matrix[self.row_clusters==j,i]
                sub_row_features = row_features[self.row_clusters==j,:]
                sub_col_features = col_features[i,:][None,:]
                X = np.concatenate([sub_row_features,np.repeat(sub_col_features,sub_row_features.shape[0],axis=0)],axis=1)
                y = sub_matrix.ravel()

                mask = fit_mask[self.row_clusters==j,i].ravel()
                y = y[mask]
                X = X[mask]

                for k in range(self.n_col_cluster): 
                    pred = self.estimators[k][j].predict(X)
                    error[k] += self.metric(y,pred)
            new_col_clusters[i] = np.argmin(error)
        return new_col_clusters
        
    def fit(self,matrix,row_features,col_features):
        
        iter_count = 0 
        changed_count = 0

        fit_mask = np.invert(np.isnan(matrix))
        
        converged = False

        self.__initialize_clustering(matrix)

        while(not converged):
            self.__fit_models(matrix,row_features,col_features,fit_mask)
            
            new_row_clusters = self.__update_row_cluster(matrix,row_features,col_features,fit_mask)
            rows_changed = np.sum(new_row_clusters==self.row_clusters)
            self.row_clusters = np.copy(new_row_clusters)

            new_col_clusters = self.__update_col_cluster(matrix,row_features,col_features,fit_mask)
            cols_changed = np.sum(new_col_clusters==self.col_clusters)
            self.col_clusters = np.copy(new_col_clusters)
            
            iter_count += 1
            if rows_changed==0 and cols_changed==0:
                changed_count+=1
            else:
                changed_count=0
            
            self.__compute_error(matrix,row_features,col_features,fit_mask)

            converged = (
                iter_count > self.max_iter or
                self.error < self.tol or
                changed_count < self.early_stop
            )   

        

        