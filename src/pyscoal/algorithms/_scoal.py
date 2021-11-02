import numpy as np
import time
from sklearn.base import clone, is_regressor
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed, Memory

class SCOAL():

    def __init__(self, 
                estimator=LinearRegression(), 
                n_row_clusters = 2, 
                n_col_clusters = 2,
                tol = 0.01,
                iter_tol = 10, 
                max_iter = 100,
                random_state=42,
                n_jobs=1,
                cache=False,
                matrix='sparse',
                verbose=False):

        self.estimator = estimator
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters
        self.tol = tol
        self.iter_tol = iter_tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cache=cache
        self.verbose = verbose
        self.matrix=matrix
        self.is_regressor = is_regressor(estimator)

    def _scoring(self,y_true,y_pred):   
        if self.is_regressor:
            return np.sum((y_true-y_pred)**2)
        else:
            return np.sum(-1*(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred)))

    def _initialize_coclusters(self,n_row_clusters,n_col_clusters):
        row_clusters = np.random.choice(np.arange(n_row_clusters),self.n_rows)
        col_clusters = np.random.choice(np.arange(n_col_clusters),self.n_cols)
        coclusters = (row_clusters, col_clusters)

        return coclusters

    def _initialize_models(self,coclusters):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        models = [[clone(self.estimator) for j in range(n_col_clusters)] 
                            for i in range(n_row_clusters)]

        return models

    def _get_rows(self,row_clusters,row_cluster):
        if row_cluster is None:
            rows_mask = np.ones(row_clusters.size).astype(bool)
        else:
            rows_mask = row_clusters==row_cluster
        
        rows = np.argwhere(rows_mask).ravel()

        return rows

    def _get_cols(self,col_clusters,col_cluster):
        if col_cluster is None:
            cols_mask = np.ones(col_clusters.size).astype(bool)
        else:
            cols_mask = col_clusters==col_cluster
        
        cols = np.argwhere(cols_mask).ravel()

        return cols

    def _get_rows_cols(self,coclusters,row_cluster,col_cluster):
        row_clusters, col_clusters = coclusters
        rows = self._get_rows(row_clusters,row_cluster)
        cols = self._get_cols(col_clusters,col_cluster)

        return rows,cols

    def _get_X_y(self,data,rows,cols):
        matrix, row_features,col_features = data

        if self.matrix=='dense':
            mask = np.invert(np.isnan(matrix)) 
            mask = mask[np.ix_(rows,cols)].ravel()
            X = np.hstack([np.repeat(row_features[rows], col_features[cols].shape[0], axis=0),
                np.tile(col_features[cols], (row_features[rows].shape[0], 1))])
            X = X[mask]
            y = matrix[np.ix_(rows,cols)].ravel()
            y = y[mask]
        else:
            rows = np.isin(matrix[:,0],rows)
            cols = np.isin(matrix[:,1],cols)
            y = matrix[rows&cols,2]
            X = np.hstack((row_features[matrix[rows&cols,0].astype(int),:],col_features[matrix[rows&cols,1].astype(int),:]))

        return X, y
    
    def _get_X(self,data,rows,cols):
        matrix, row_features,col_features = data

        if self.matrix=='dense':
            mask = np.invert(np.isnan(matrix)) 
            mask = mask[np.ix_(rows,cols)].ravel()
            X = np.hstack([np.repeat(row_features[rows], col_features[cols].shape[0], axis=0),
                np.tile(col_features[cols], (row_features[rows].shape[0], 1))])
            X = X[mask]
        else:
            rows = np.isin(matrix[:,0],rows)
            cols = np.isin(matrix[:,1],cols)
            X = np.hstack((row_features[matrix[rows&cols,0].astype(int),:],col_features[matrix[rows&cols,1].astype(int),:]))

        return X

    def _get_y(self,data,rows,cols):
        matrix, _ , _ = data

        if self.matrix=='dense':
            mask = np.invert(np.isnan(matrix)) 
            mask = mask[np.ix_(rows,cols)].ravel()
            y = matrix[np.ix_(rows,cols)].ravel()
            y = y[mask]
        else:
            rows = np.isin(matrix[:,0],rows)
            cols = np.isin(matrix[:,1],cols)
            y = matrix[rows&cols,2]

        return y
    
    def _check(self,data,coclusters,models,row_cluster,col_cluster):
        rows,cols = self._get_rows_cols(coclusters,row_cluster,col_cluster)
        y = self._get_y(data,rows,cols)
        checked = y.size

        return checked

    def _cached_fit(self,model,X,y,rows,cols):
        model.fit(X,y)
        del X
        del y
        
        return model
    
    def _sklearn_fit(self,model,X,y):
        model.fit(X,y)

        return model

    def _fit(self,data,coclusters,models,row_cluster,col_cluster):
        rows,cols = self._get_rows_cols(coclusters,row_cluster,col_cluster)
        X, y = self._get_X_y(data,rows,cols)
        model = models[row_cluster][col_cluster]
        if y.size > 0:
            if self.cache:
                model = self._cached_fit(model,X,y,rows,cols)
            else:
                model = self._sklearn_fit(model,X,y)
        del X
        del y

        return model

    def _fit_predict(self,data,coclusters,models,row_cluster,col_cluster):
        rows,cols = self._get_rows_cols(coclusters,row_cluster,col_cluster)
        X, y = self._get_X_y(data,rows,cols)
        model = models[row_cluster][col_cluster]
        y_pred = np.nan
        if X.size >0:
            if self.cache:
                model = self._cached_fit(model,X,y,rows,cols)
            else:
                model = self._sklearn_fit(model,X,y)
            y_pred = model.predict(X) if self.is_regressor else model.predict_proba(X)[:,1]

        return model, y_pred 

    def _fit_score(self,data,coclusters,models,row_cluster,col_cluster):
        rows,cols = self._get_rows_cols(coclusters,row_cluster,col_cluster)
        X, y = self._get_X_y(data,rows,cols)
        model = models[row_cluster][col_cluster]
        score = 0
        if y.size > 0:
            if self.cache:
                model = self._cached_fit(model,X,y,rows,cols)
            else:
                model = self._sklearn_fit(model,X,y)
            y_pred = model.predict(X) if self.is_regressor else model.predict_proba(X)[:,1]
            score = self._scoring(y,y_pred)
        del X
        del y

        return model, score  
    
    def _predict(self,data,coclusters,models,row_cluster,col_cluster):
        rows,cols = self._get_rows_cols(coclusters,row_cluster,col_cluster)
        X = self._get_X(data,rows,cols)
        model = models[row_cluster][col_cluster]
        y_pred = np.nan
        if X.size >0:
            y_pred = model.predict(X)
        del X

        return y_pred 
    
    def _score(self,data,coclusters,models,row_cluster,col_cluster):
        rows,cols = self._get_rows_cols(coclusters,row_cluster,col_cluster)
        X, y = self._get_X_y(data,rows,cols)
        model = models[row_cluster][col_cluster]
        score = 0
        if y.size > 0:
            y_pred = model.predict(X) if self.is_regressor else model.predict_proba(X)[:,1]
            score = self._scoring(y,y_pred)
        del X
        del y

        return score 
    
    def _score_rows(self,data,coclusters,models,row_cluster,col_cluster):
        rows,cols = self._get_rows_cols(coclusters,None,col_cluster)
        model = models[row_cluster][col_cluster]
        scores = np.zeros(rows.size)
        for row in rows:
            X, y = self._get_X_y(data,[row],cols)
            if y.size > 0:
                y_pred = model.predict(X) if self.is_regressor else model.predict_proba(X)[:,1]
                scores[row] = self._scoring(y,y_pred)
        del X
        del y

        return scores
    
    def _score_cols(self,data,coclusters,models,row_cluster,col_cluster):
        rows,cols = self._get_rows_cols(coclusters,row_cluster,None)
        model = models[row_cluster][col_cluster]
        scores = np.zeros(cols.size)
        for col in cols:
            X, y = self._get_X_y(data,rows,[col])
            if y.size > 0:
                y_pred = model.predict(X) if self.is_regressor else model.predict_proba(X)[:,1]
                scores[col] = self._scoring(y,y_pred)
        del X
        del y

        return scores

    def _compute_clusterwise(self,data,coclusters,models,function,n_jobs):    
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size

        results = Parallel(n_jobs=n_jobs,backend='loky')(delayed(function)
            (data,coclusters,models,i,j) for i in range(n_row_clusters) for j in range(n_col_clusters))
        results = [[results[i*n_col_clusters+j]
            for j in range(n_col_clusters)] for i in range(n_row_clusters)]

        return results

    def _update_models(self,data,coclusters,models,n_jobs):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        results = self._compute_clusterwise(data,coclusters,models,self._fit_score,n_jobs)
        models =  [[results[i][j][0]
            for j in range(n_col_clusters)] for i in range(n_row_clusters)]
        scores = np.array([[results[i][j][1]
            for j in range(n_col_clusters)] for i in range(n_row_clusters)])

        return models, scores

    def _update_row_clusters(self,data,coclusters,models,n_jobs):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        results = self._compute_clusterwise(data,coclusters,models,self._score_rows,n_jobs)
        scores = np.zeros((row_clusters.size,n_row_clusters))
        for i in range(n_row_clusters):
            for j in range(n_col_clusters):
                scores[:,i] += results[i][j]
        new_row_clusters  = np.argmin(scores,axis=1)

        return new_row_clusters

    def _update_col_clusters(self,data,coclusters,models,n_jobs):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        results = self._compute_clusterwise(data,coclusters,models,self._score_cols,n_jobs)
        scores = np.zeros((col_clusters.size,n_col_clusters))
        for i in range(n_row_clusters):
            for j in range(n_col_clusters):
                scores[:,j] += results[i][j] 
        new_col_clusters  = np.argmin(scores,axis=1)

        return new_col_clusters

    def _update_coclusters(self,data,coclusters,models,n_jobs):
        new_row_clusters = self._update_row_clusters(data,coclusters,models,n_jobs)
        new_col_clusters = self._update_col_clusters(data,coclusters,models,n_jobs)

        return new_row_clusters,new_col_clusters

    def _check_coclusters(self,data,coclusters,models,n_jobs):
        results = self._compute_clusterwise(data,coclusters,models,self._check,n_jobs)
        checked = np.array(results)
        
        return checked

    def _fit_coclusters(self,data,coclusters,models,n_jobs):
        results = self._compute_clusterwise(data,coclusters,models,self._fit,n_jobs)
        models = results
        
        return models

    def _predict_coclusters(self,data,coclusters,models,n_jobs):
        results = self._compute_clusterwise(data,coclusters,models,self._predict,n_jobs)
        predictions = np.array(results)
        
        return predictions

    def _score_coclusters(self,data,coclusters,models,n_jobs):
        results = self._compute_clusterwise(data,coclusters,models,self._score,n_jobs)
        scores = np.array(results)
        
        return scores

    def _log(self,iter_count,score,delta_score,rows_changed,cols_changed,elapsed_time):
        if iter_count==0:
            print('|'.join(x.ljust(15) for x in [
                    'iteration',' score','delta score (%)','rows changed', 'columns changed', 'elapsed time (s)']))

        print('|'.join(x.ljust(15) for x in ['%i' % iter_count,'%.4f' % score,'%.4f' % delta_score,'%i' % rows_changed,'%i'  % cols_changed,'%i' % elapsed_time]))
    
    def _converge_scoal(self,data,coclusters,models,tol=0.01,iter_tol=10,max_iter=10,n_jobs=1,verbose=False):
        iter_count=0 
        elapsed_time = 0
        rows_changed = 0
        cols_changed = 0
        score = np.nan
        delta_score=np.nan
        delta_scores=np.ones(iter_tol)
        converged = False
        start = time.time()

        if coclusters is None:
            coclusters = self._initialize_coclusters(self.n_row_clusters,self.n_col_clusters)
        
        if models is None:
            models = self._initialize_models(coclusters)

        models, scores = self._update_models(data,coclusters,models,n_jobs)
        score = np.sum(scores)/self.n_values

        if verbose:
            self._log(iter_count,score,delta_score,rows_changed,cols_changed,elapsed_time)
        
        converged = iter_count == max_iter 
        
        while not converged:
            new_row_clusters, new_col_clusters = self._update_coclusters(data,coclusters,models,n_jobs)     
            rows_changed = np.sum(new_row_clusters!=coclusters[0])
            cols_changed = np.sum(new_col_clusters!=coclusters[1])
            coclusters = (new_row_clusters, new_col_clusters)
            old_score = score
            models, scores = self._update_models(data,coclusters,models,n_jobs)
            score = np.sum(scores)/self.n_values
            delta_score = (old_score-score)/old_score
            delta_scores[iter_count%iter_tol] = delta_score
            iter_count += 1
            converged = (
                iter_count >= max_iter or
                (np.max(delta_scores) < tol) or
                (rows_changed==0 and cols_changed==0)
            )   
            elapsed_time = time.time() - start
            if verbose:
                self._log(iter_count,score,delta_score,rows_changed,cols_changed,elapsed_time)

        return coclusters, models 

    def fit(self,target,row_features,col_features,coclusters=None):
        np.random.seed(self.random_state) 

        self.n_rows, self.n_cols, self.n_values = row_features.shape[0], col_features.shape[0], target.shape[0]
        self.n_row_features, self.n_col_features  = row_features.shape[1], col_features.shape[1]        
        if self.matrix=='dense':
            matrix = np.zeros((self.n_rows, self.n_cols))*np.nan
            matrix[target[:,0].astype(int),target[:,1].astype(int)] = target[:,2]
        else:
            matrix = target 
        del target

        data = (matrix,row_features,col_features)
   
        if self.cache:
            self.memory = Memory('./pyscoal-cache')
            self._cached_fit = self.memory.cache(self._cached_fit, ignore=['self','model','X','y'])
     
        self.coclusters,self.models = self._converge_scoal(data,coclusters,None,self.tol,self.iter_tol,self.max_iter,self.n_jobs,self.verbose)

        if self.cache:
            self.memory.clear(warn=False)

    def predict(self,target,row_features,col_features):
        n_rows, n_cols, n_values = row_features.shape[0], col_features.shape[0], target.shape[0]
        n_row_features, n_col_features  = row_features.shape[1], col_features.shape[1]    
        if self.matrix=='dense':
            rows, cols = target[:,0].astype(int), target[:,1].astype(int)
            matrix = np.zeros((self.n_rows, self.n_cols))*np.nan
            matrix[rows,cols] = 0
            sorting = np.lexsort((target[:, 1],target[:, 0]))
        else:
            matrix = np.hstack((target,np.zeros((target.shape[0],1))))  
            sorting = np.arange(target.shape[0])

        data = (matrix,row_features,col_features)     

        coclusters_predictions = self._predict_coclusters(data,self.coclusters,self.models,self.n_jobs)
        predictions = np.zeros(n_values,dtype='float64')

        row_clusters, col_clusters = self.coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        for row_cluster in range(n_row_clusters):
            for col_cluster in range(n_col_clusters):
                    rows = row_clusters[target[sorting,0].astype(int)]==row_cluster
                    cols = col_clusters[target[sorting,1].astype(int)]==col_cluster
                    predictions[rows&cols] = coclusters_predictions[row_cluster][col_cluster]
        
        predictions = predictions[np.argsort(sorting)]

        return predictions
    
    def score(self,target,row_features,col_features):
        n_rows, n_cols, n_values = row_features.shape[0], col_features.shape[0], target.shape[0]
        n_row_features, n_col_features  = row_features.shape[1], col_features.shape[1]    

        if self.matrix=='dense':
                rows, cols, values = target[:,0].astype(int), target[:,1].astype(int), target[:,2]
                matrix = np.zeros((self.n_rows, self.n_cols))*np.nan
                matrix[rows,cols] = values
        else:
            matrix = target 

        data = (matrix,row_features,col_features)        

        coclusters_scores = self._score_coclusters(data,self.coclusters,self.models,self.n_jobs)

        score = np.sum(coclusters_scores)/n_values  

        return score 