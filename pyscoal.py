import numpy as np
import time
from sklearn.base import clone
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted


class BaseScoal():

    def __init__(self,estimator=LinearRegression(),scoring=mean_squared_error,minimize=True,init='random',random_state=42,n_jobs=1):
        self.estimator = estimator()
        self.scoring=mean_squared_error
        self.minimize=minimize
        self.init='smart'
        self.n_jobs=1

    def _random_init(self,mask,n_row_clusters,n_col_clusters):
        n_rows, n_cols = mask.shape
        row_clusters = np.random.choice(np.arange(n_row_clusters),n_rows)
        col_clusters = np.random.choice(np.arange(n_col_clusters),n_cols)
        
        return row_clusters, col_clusters 
        
    def _smart_init(self,mask,n_row_clusters,n_col_clusters):
        n_rows, n_cols = mask.shape
        row_clusters = np.zeros(n_rows)*np.nan
        col_clusters = np.zeros(n_cols)*np.nan
        row_clusters_aux = np.zeros((n_row_clusters,n_cols))
        for row in np.arange(n_rows):
            row_cluster = np.argmax((mask[row,:]>row_clusters_aux).sum(axis=1))
            row_clusters_aux[row_cluster]=np.logical_or(row_clusters_aux[row_cluster],mask[row,:])
            row_clusters[row] = row_cluster
        col_clusters_aux = np.zeros((n_col_clusters,n_row_clusters))
        for col in np.arange(n_cols):
            col_cluster = np.argmax((row_clusters_aux[:,col]>col_clusters_aux).sum(axis=1))
            col_clusters_aux[col_cluster]=np.logical_or(col_clusters_aux[col_cluster],row_clusters_aux[:,col])
            col_clusters[col] = col_cluster
    
        return row_clusters.astype(int),col_clusters.astype(int)

    def _initialize_coclusters(self,mask,n_row_clusters,n_col_clusters):
        if self.init=='smart':
            row_clusters, col_clusters = self._smart_init(mask,n_row_clusters,n_col_clusters)
        elif isinstance(self.init,(list,tuple,np.ndarray)):
            row_clusters, col_clusters = self.init[0], self.init[1]
        else:
            row_clusters, col_clusters = self._random_init(mask,n_row_clusters,n_col_clusters)
        coclusters = (row_clusters, col_clusters)

        return coclusters

    def _initialize_models(self,mask,coclusters):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        models = [[clone(self.estimator) for j in range(n_col_clusters)] 
                            for i in range(n_row_clusters)]

        return models

    def _check_coclusters(self,mask,coclusters):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        valid = np.zeros((n_row_clusters,n_col_clusters)).astype(bool)
        for i in range(n_row_clusters):
            for j in range(n_col_clusters):
                rows,cols = self._get_rows_cols(coclusters,i,j)
                valid[i,j] = mask[np.ix_(rows,cols)].any()
        
        return valid

    def _get_rows_cols(self,coclusters,row_cluster,col_cluster):
        row_clusters, col_clusters = coclusters
        if row_cluster is None:
            rows_mask = np.ones(row_clusters.size).astype(bool)
        else:
            rows_mask = row_clusters==row_cluster
        if col_cluster is None:
            cols_mask = np.ones(col_clusters.size).astype(bool)
        else:
            cols_mask = col_clusters==col_cluster
        rows = np.argwhere(rows_mask).ravel()
        cols = np.argwhere(cols_mask).ravel()

        return rows,cols

    def _get_bool_mask(self,coclusters,row_cluster,col_cluster):
        row_clusters, col_clusters = coclusters
        if row_cluster is None:
            rows_mask = np.ones(row_clusters.size).astype(bool)
        else:
            rows_mask = row_clusters==row_cluster
        if col_cluster is None:
            cols_mask = np.ones(col_clusters.size).astype(bool)
        else:
            cols_mask = col_clusters==col_cluster
        cocluster_mask = np.logical_and(
            np.repeat(rows_mask.reshape(-1,1),cols_mask.shape,axis=1),
            np.repeat(cols_mask.reshape(1,-1),rows_mask.shape,axis=0)
        )
        
        return cocluster_mask

    def _get_X_y(self,data,mask,rows,cols):
        matrix, row_features,col_features = data
        mask = mask[np.ix_(rows,cols)].ravel()
        X = np.hstack([np.repeat(row_features[rows], col_features[cols].shape[0], axis=0),
            np.tile(col_features[cols], (row_features[rows].shape[0], 1))])
        X = X[mask]
        y = matrix[np.ix_(rows,cols)].ravel()
        y = y[mask]

        #idx = np.where(mask & cocluster_mask)
        #X = np.hstack((row_features[idx[0]],col_features[idx[1]]))
        #y = matrix[idx].ravel()

        return X, y

    def _fit(self,data,mask,coclusters,models,row_cluster,col_cluster):
        rows,cols = self._get_rows_cols(coclusters,row_cluster,col_cluster)
        X, y = self._get_X_y(data,mask,rows,cols)
        model = models[row_cluster][col_cluster]
        model.fit(X,y)

        return model

    def _fit_predict(self,data,mask,coclusters,models,row_cluster,col_cluster):
        rows,cols = self._get_rows_cols(coclusters,row_cluster,col_cluster)
        X, y = self._get_X_y(data,mask,rows,cols)
        model = models[row_cluster][col_cluster]
        model.fit(X,y)
        y_pred = model.predict(X)

        return model, y_pred 

    def _fit_score(self,data,mask,coclusters,models,row_cluster,col_cluster):
        rows,cols = self._get_rows_cols(coclusters,row_cluster,col_cluster)
        X, y = self._get_X_y(data,mask,rows,cols)
        model = models[row_cluster][col_cluster]
        model.fit(X,y)
        y_pred = model.predict(X)
        score = self.scoring(y,y_pred)

        return model, score  
    
    def _predict(self,data,mask,coclusters,models,row_cluster,col_cluster):
        rows,cols = self._get_rows_cols(coclusters,row_cluster,col_cluster)
        X,_ = self._get_X_y(data,mask,rows,cols)
        model = models[row_cluster][col_cluster]
        y_pred = model.predict(X)

        return y_pred 
    
    def _score(self,data,mask,coclusters,models,row_cluster,col_cluster):
        rows,cols = self._get_rows_cols(coclusters,row_cluster,col_cluster)
        X, y = self._get_X_y(data,mask,rows,cols)
        model = models[row_cluster][col_cluster]
        score = 0
        if y.size>0:
            y_pred = model.predict(X)
            score = self.scoring(y,y_pred)

        return score 
    
    def _score_rows(self,data,mask,coclusters,models,row_cluster,col_cluster):
        rows,cols = self._get_rows_cols(coclusters,None,col_cluster)
        model = models[row_cluster][col_cluster]
        scores = np.zeros(rows.size)
        for row in rows:
            X, y = self._get_X_y(data,mask,[row],cols)
            if y.size>0:
                y_pred = model.predict(X)
                scores[row] = self.scoring(y,y_pred)

        return scores
    
    def _score_cols(self,data,mask,coclusters,models,row_cluster,col_cluster):
        rows,cols = self._get_rows_cols(coclusters,row_cluster,None)
        model = models[row_cluster][col_cluster]
        scores = np.zeros(cols.size)
        for col in cols:
            X, y = self._get_X_y(data,mask,rows,[col])
            if(y.size>0):
                y_pred = model.predict(X)
                scores[col] = self.scoring(y,y_pred)

        return scores
        
    def _compute_parallel(self,data,mask,coclusters,models,function):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        results = Parallel(n_jobs=self.n_jobs)(delayed(function)
            (data,mask,coclusters,models,i,j) 
            for i in range(n_row_clusters) for j in range(n_col_clusters))

        return results

    def _update_models(self,data,mask,coclusters,models):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        results = self._compute_parallel(data,mask,coclusters,models,self._fit_score)
        models =  [[results[i*n_col_clusters+j][0]
            for j in range(n_col_clusters)] for i in range(n_row_clusters)]
        scores = np.array([[results[i*n_col_clusters+j][1]
            for j in range(n_col_clusters)] for i in range(n_row_clusters)])

        return models, scores

    def _update_row_clusters(self,data,mask,coclusters,models):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        results = self._compute_parallel(data,mask,coclusters,models,self._score_rows)
        scores = np.zeros((row_clusters.size,n_row_clusters))
        for i in range(n_row_clusters):
            for j in range(n_col_clusters):
                scores[:,i] += results[i*n_col_clusters+j] 
        scores = scores/n_col_clusters
        new_row_clusters  = np.argmin(scores,axis=1) if self.minimize else np.argmax(scores,axis=1)

        return new_row_clusters
    
    def _update_col_clusters(self,data,mask,coclusters,models):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        results = self._compute_parallel(data,mask,coclusters,models,self._score_cols)
        scores = np.zeros((col_clusters.size,n_col_clusters))
        for i in range(n_row_clusters):
            for j in range(n_col_clusters):
                scores[:,j] += results[i*n_col_clusters+j] 
        scores = scores/n_row_clusters
        new_col_clusters  = np.argmin(scores,axis=1) if self.minimize else np.argmax(scores,axis=1)

        return new_col_clusters

    def _update_coclusters(self,data,mask,coclusters,models):
        new_row_clusters = self._update_row_clusters(data,mask,coclusters,models)
        new_col_clusters = self._update_col_clusters(data,mask,coclusters,models)

        return new_row_clusters,new_col_clusters

    def _fit_coclusters(self,data,mask,coclusters,models):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        results = self._compute_parallel(data,mask,coclusters,models,self._fit)
        models =  [[results[i*n_col_clusters+j]
            for j in range(n_col_clusters)] for i in range(n_row_clusters)]
        
        return models

    def _predict_coclusters(self,data,mask,coclusters,models):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        results = self._compute_parallel(data,mask,coclusters,models,self._predict)
        predictions =  [[results[i*n_col_clusters+j]
            for j in range(n_col_clusters)] for i in range(n_row_clusters)]
        
        return predictions

    def _score_coclusters(self,data,mask,coclusters,models):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        results = self._compute_parallel(data,mask,coclusters,models,self._score)
        scores =  [[results[i*n_col_clusters+j]
            for j in range(n_col_clusters)] for i in range(n_row_clusters)]
        
        return scores


class SCOAL(BaseScoal):
    
    def __init__(self, 
                estimator=LinearRegression(), 
                n_row_clusters = 2, 
                n_col_clusters = 2,
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

    def _print_status(self,iter_count,score,delta_score,rows_changed,cols_changed,elapsed_time):
        if iter_count==0:
            print('|'.join(x.ljust(15) for x in [
                    'iteration',' score','delta score','rows changed', 'columns changed', 'elapsed time (s)']))

        print('|'.join(x.ljust(15) for x in ['%i' % iter_count,'%.3f' % score,'%.3f' % delta_score,'%i' % rows_changed,'%i'  % cols_changed,'%i' % elapsed_time]))

    def fit(self,matrix,row_features,col_features,fit_mask=None):
        np.random.seed(self.random_state) 
        data = (matrix,row_features,col_features)
        if fit_mask is None:
            fit_mask = np.invert(np.isnan(matrix))         

        iter_count=0 
        elapsed_time = 0
        rows_changed = 0
        cols_changed = 0
        score = np.nan
        delta_score=np.nan
        converged = False
        start = time.time()

        self.coclusters = self._initialize_coclusters(fit_mask,self.n_row_clusters,self.n_col_clusters)
        self.models = self._initialize_models(fit_mask,self.coclusters)

        self.models, self.scores = self._update_models(data,fit_mask,self.coclusters,self.models)
        score = np.mean(self.scores)
        
        converged = iter_count == self.max_iter 
        if self.verbose:
            self._print_status(iter_count,score,delta_score,rows_changed,cols_changed,elapsed_time)

        while not converged:

            new_row_clusters, new_col_clusters = self._update_coclusters(data,fit_mask,self.coclusters,self.models)     
            rows_changed = np.sum(new_row_clusters!=self.coclusters[0])
            cols_changed = np.sum(new_col_clusters!=self.coclusters[1])
            self.coclusters = (np.copy(new_row_clusters), np.copy(new_col_clusters))
            delta_score = score
            self.models, self.scores = self._update_models(data,fit_mask,self.coclusters,self.models)
            score = np.mean(self.scores)            
            delta_score -= score
            iter_count += 1
            converged = (
                iter_count >= self.max_iter or
                (delta_score > 0 and delta_score < self.tol) or
                (rows_changed==0 and cols_changed==0)
            )   
            elapsed_time = time.time() - start
            if self.verbose:
                self._print_status(iter_count,score,delta_score,rows_changed,cols_changed,elapsed_time)
        self.n_iter = iter_count
        
    def predict(self,matrix,row_features,col_features,pred_mask=None):
        data = (matrix,row_features,col_features)
        if pred_mask is None:
            pred_mask = np.isnan(matrix)
        pred_matrix = matrix.copy()
        results = self._predict_coclusters(data,pred_mask,self.coclusters,self.models)
        for i in range(self.n_row_clusters):
            for j in range(self.n_col_clusters):
                cocluster_mask = self._get_bool_mask(self.coclusters,i,j)
                pred_matrix[cocluster_mask&pred_mask] = results[i][j]

        return pred_matrix
    
    def score(self,matrix,row_features,col_features,pred_mask=None):
        data = (matrix,row_features,col_features)
        if pred_mask is None:
            pred_mask = np.invert(np.isnan(matrix))
        scores = self._score_coclusters(data,pred_mask,self.coclusters,self.models)
        score = np.mean(np.array(scores))

        return score


class EvolutiveScoal(BaseScoal):
    
    def __init__(self,
                max_row_clusters=10,
                max_col_clusters=10,
                pop_size=20,
                estimator=LinearRegression(),
                scoring=mean_squared_error,
                minimize=True,
                test_size=0.2,
                max_iter=np.nan,
                tol = 1e-4,
                init='random',
                random_state=42,
                n_jobs=1,
                verbose=False):

        self.max_row_clusters=max_row_clusters
        self.max_col_clusters=max_col_clusters
        self.pop_size = pop_size
        self.estimator=estimator
        self.scoring=mean_squared_error
        self.minimize=minimize
        self.test_size=test_size
        self.max_iter = max_iter
        self.tol=tol
        self.init=init
        self.random_state=random_state
        self.n_jobs=n_jobs
        self.verbose=verbose

    def _local_search(self,data,fit_mask,pop):
        new_pop = []
        for ind in pop:
            if not np.all(self._check_coclusters(fit_mask,ind)):
                continue
            models = self._initialize_models(fit_mask,ind)
            models,_ = self._update_models(data,fit_mask,ind,models)
            new_row_clusters,new_col_clusters = self._update_coclusters(data,fit_mask,ind,models)
            new_pop.append((new_row_clusters,new_col_clusters))

        return new_pop  
        
    def _delete_cluster(self,clusters,cluster):
        n_clusters = np.unique(clusters).size
        if n_clusters > 1:
            clusters[clusters==cluster] = -1
            clusters[clusters>cluster] -= 1
            clusters[clusters==-1] = np.random.choice(np.arange(n_clusters-1),(clusters==-1).sum())

        return clusters

    def _delete_row_cluster(self,mask,coclusters,row_cluster):
        row_clusters,col_clusters = coclusters
        n_row_clusters, n_col_clusters = np.unique(row_clusters).size, np.unique(col_clusters).size
        if n_row_clusters == 1:
            return row_clusters
        else:
            n_rows, n_cols = mask.shape
            new_row_clusters = np.zeros(n_rows)*np.nan
            col_clusters_aux = np.zeros((n_col_clusters,n_rows))
            row_clusters_aux = np.zeros((n_row_clusters-1,n_col_clusters))
            row_clusters[row_clusters==row_cluster] = -1
            row_clusters[row_clusters>row_cluster] -= 1
            for col in np.arange(n_cols):
                col_cluster = col_clusters[col]
                col_clusters_aux[col_cluster]=np.logical_or(col_clusters_aux[col_cluster],mask[:,col])  
            for row in np.arange(n_rows):
                if row_clusters[row]==-1:
                    new_row_cluster = np.argmax((col_clusters_aux[:,row]>row_clusters_aux).sum(axis=1))
                else:
                    new_row_cluster=row_clusters[row]
                row_clusters_aux[new_row_cluster]=np.logical_or(row_clusters_aux[new_row_cluster],col_clusters_aux[:,row])
                new_row_clusters[row] = new_row_cluster
        
        return new_row_clusters

    def _delete_col_cluster(self,mask,coclusters,col_cluster):
        row_clusters,col_clusters = coclusters
        n_row_clusters, n_col_clusters = np.unique(row_clusters).size, np.unique(col_clusters).size
        if n_col_clusters == 1:
            return col_cluster
        else:
            n_rows, n_cols = mask.shape
            new_col_clusters = np.zeros(n_cols)*np.nan
            row_clusters_aux = np.zeros((n_row_clusters,n_cols))
            col_clusters_aux = np.zeros((n_col_clusters-1,n_row_clusters))
            col_clusters[col_clusters==col_cluster] = -1
            col_clusters[col_clusters>col_cluster] -= 1
            for row in np.arange(n_rows):
                row_cluster = row_clusters[row]
                row_clusters_aux[row_cluster]=np.logical_or(row_clusters_aux[row_cluster],mask[row,:])
            for col in np.arange(n_cols):
                if col_clusters[col]==-1:
                    new_col_cluster = np.argmax((row_clusters_aux[:,col]>col_clusters_aux).sum(axis=1))
                else:
                    new_col_cluster=col_clusters[col]
                col_clusters_aux[new_col_cluster]=np.logical_or(col_clusters_aux[new_col_cluster],row_clusters_aux[:,col])
                new_col_clusters[col] = new_col_cluster
        
        return new_col_clusters

    def _split_row_cluster(self,mask,coclusters,row_cluster):
        row_clusters,col_clusters = coclusters
        n_row_clusters, n_col_clusters = np.unique(row_clusters).size, np.unique(col_clusters).size
        if n_row_clusters == self.max_row_clusters:
            return row_clusters
        else:
            n_rows, n_cols = mask.shape
            new_row_clusters = np.zeros(n_rows)*np.nan
            col_clusters_aux = np.zeros((n_col_clusters,n_rows))
            row_clusters_aux = np.zeros((n_row_clusters+1,n_col_clusters))
            for col in np.arange(n_cols):
                col_cluster = col_clusters[col]
                col_clusters_aux[col_cluster]=np.logical_or(col_clusters_aux[col_cluster],mask[:,col])
            for row in np.arange(n_rows):
                if row_clusters[row]==row_cluster:
                    new_row_cluster = np.argmax((col_clusters_aux[:,row]>row_clusters_aux[[row_cluster,n_row_clusters],:]).sum(axis=1))
                else:
                    new_row_cluster=row_clusters[row]
                row_clusters_aux[new_row_cluster]=np.logical_or(row_clusters_aux[new_row_cluster],col_clusters_aux[:,row])
                new_row_clusters[row] = new_row_cluster

        return new_row_clusters
    
    def _split_col_cluster(self,mask,coclusters,col_cluster):
        row_clusters,col_clusters = coclusters
        n_row_clusters, n_col_clusters = np.unique(row_clusters).size, np.unique(col_clusters).size
        if n_col_clusters == self.max_col_clusters:
            return row_clusters
        else:
            n_rows, n_cols = mask.shape
            new_col_clusters = np.zeros(n_cols)*np.nan
            row_clusters_aux = np.zeros((n_row_clusters,n_cols))
            col_clusters_aux = np.zeros((n_col_clusters+1,n_row_clusters))
            for row in np.arange(n_rows):
                row_cluster = row_clusters[row]
                row_clusters_aux[row_cluster]=np.logical_or(row_clusters_aux[row_cluster],mask[row,:])
            for col in np.arange(n_cols):
                if col_clusters[col]==col_cluster:
                    new_col_cluster = np.argmax((row_clusters_aux[:,col]>col_clusters_aux[[col_cluster,n_col_clusters],:]).sum(axis=1))
                else:
                    new_col_cluster=col_clusters[col]
                col_clusters_aux[new_col_cluster]=np.logical_or(col_clusters_aux[new_col_cluster],row_clusters_aux[:,col])
                new_col_clusters[col] = new_col_cluster

        return new_col_clusters

    def _mutation(self,mask,pop,fitness):
        new_pop = []
        for i, ind in enumerate(pop):
            if not np.all(self._check_coclusters(mask,ind)):
                continue
            new_ind = deepcopy(ind)
            new_row_clusters,new_col_clusters = new_ind
            n_row_clusters, n_col_clusters = np.unique(new_row_clusters).size, np.unique(new_col_clusters).size
            all_fitness = np.concatenate((np.nanmean(fitness[i],axis=1),np.nanmean(fitness[i],axis=0)),axis=0)
            probs = np.divide(1,all_fitness, where=all_fitness!=0) if not self.minimize else all_fitness
            probs[probs==0] = 1
            probs = np.nan_to_num(probs)
            probs = probs/probs.sum()
            choice = np.random.choice(np.arange(probs.size),p=probs)
            if choice < self.max_row_clusters:
                row_cluster=choice
                if abs(n_row_clusters - self.max_row_clusters) < abs(n_row_clusters - 1):
                    new_row_clusters=self._delete_row_cluster(mask,new_ind,row_cluster)
                else:
                    new_row_clusters=self._split_row_cluster(mask,new_ind,row_cluster)

            else:
                col_cluster=choice-self.max_row_clusters
                if abs(n_col_clusters - self.max_row_clusters) < abs(n_col_clusters - 1):
                    new_col_clusters=self._delete_col_cluster(mask,new_ind,col_cluster)
                else:
                    new_col_clusters=self._split_col_cluster(mask,new_ind,col_cluster)
            new_pop.append((new_row_clusters,new_col_clusters))

        return new_pop
   
    def _replacement(self,pop,fitness,new_pop,new_fitness):
        all_fitness = np.nanmean(np.concatenate((fitness,new_fitness),axis=0),axis=(1,2))
        best = np.nanargmin(all_fitness) if self.minimize else np.nanargmax(all_fitness)
        probs = np.divide(1,all_fitness, where=all_fitness!=0) if self.minimize else all_fitness
        probs[probs==0] = 1
        probs = np.nan_to_num(probs)
        probs[best] = 0.
        probs = probs/probs.sum()
        choice = np.random.choice(np.arange(probs.size),self.pop_size-1,replace=False,p=probs)
        pop = pop + new_pop
        new_pop = [pop[i] for i in choice]
        new_pop.append(pop[best])
        fitness = np.concatenate((fitness,new_fitness),axis=0)
        new_fitness = fitness[[best]+choice.tolist(),:,:]

        return new_pop, new_fitness
            
    def _evaluate_fitness(self,data,fit_mask,test_mask,pop):
        fitness = np.zeros((self.pop_size,self.max_row_clusters,self.max_col_clusters))*np.nan
        for i,ind in enumerate(pop):
            row_clusters,col_clusters = ind
            n_row_clusters, n_col_clusters = np.unique(row_clusters).size, np.unique(col_clusters).size
            if np.all(self._check_coclusters(fit_mask,ind)):
                models = self._initialize_models(fit_mask,ind)
                models, _ = self._update_models(data,fit_mask,ind,models)
                scores = self._score_coclusters(data,test_mask,ind,models)
                fitness[i,:n_row_clusters,:n_col_clusters] = scores
        return fitness
       
    def _init_population(self,fit_mask):
        n_row_clusters = np.random.randint(1,self.max_row_clusters+1,self.pop_size)
        n_col_clusters = np.random.randint(1,self.max_col_clusters+1,self.pop_size)
        pop = [(self._initialize_coclusters(fit_mask,i,j)) for i,j in zip(n_row_clusters,n_col_clusters)]

        return pop

    def _check_population(self,mask,pop):
        valid = np.zeros(self.pop_size).astype(bool)
        for i,ind in enumerate(pop):
            valid[i] = (self._check_coclusters(mask,ind)).all()

        return valid

    def _print_status(self,iter_count,pop,fitness,delta_score,elapsed_time):
        scores = np.nanmean(fitness,axis=(1,2))
        best_score = scores.min() if self.minimize else scores.max()
        worst_score = scores.max() if self.minimize else scores.min()
        mean_score = scores.mean()
        sizes = np.array([(np.max(ind[0])+1)*(np.max(ind[1])+1) for ind in pop])
        max_size = sizes.max()
        min_size = sizes.min()
        mean_size = sizes.mean()
        if iter_count==0:
            print('|'.join(x.ljust(11) for x in [
                    'iteration','delta score','best score','worst score', 'mean score','max size','min size','mean size', 'elapsed time (s)']))

        print('|'.join(x.ljust(11) for x in ['%i' % iter_count,'%.3f' % delta_score,'%.3f' % best_score,'%.3f' % worst_score,'%.3f'  % mean_score,'%i'  % max_size,'%i'  % min_size,'%i'  % mean_size, '%i' % elapsed_time]))

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

        iter_count = 0
        converged = False
        delta_score = np.nan
        elapsed_time = 0
        start = time.time()

        pop = self._init_population(fit_mask)
        valid_solutions = self._check_population(fit_mask,pop)
        if not np.all(valid_solutions):
            print('invalid solution at initialization')
        fitness = self._evaluate_fitness(data,fit_mask,test_mask,pop)
        self.pop = pop
        self.fitness=fitness
        score = np.nanmean(fitness,axis=(1,2)).min() if self.minimize else np.nanmean(fitness,axis=(1,2)).max()
        converged = (
                iter_count == self.max_iter or 
                (delta_score > 0 and delta_score < self.tol)
            )
        if self.verbose:
            self._print_status(iter_count,pop,fitness,delta_score,elapsed_time)
        
        while not converged:
        
            pop = self._local_search(data,fit_mask,pop)
            valid_solutions = self._check_population(fit_mask,pop)
            if not np.all(valid_solutions):
                print('invalid solution at scoal')
            fitness = self._evaluate_fitness(data,fit_mask,test_mask,pop)
            new_pop = self._mutation(fit_mask,pop,fitness)
            new_fitness = self._evaluate_fitness(data,fit_mask,test_mask,new_pop)
            pop,fitness = self._replacement(pop,fitness,new_pop,new_fitness)
            self.pop = pop
            self.fitness=fitness
            delta_score = score
            score = np.nanmean(fitness,axis=(1,2)).min() if self.minimize else np.nanmean(fitness,axis=(1,2)).max()
            delta_score -= score
            iter_count+=1
            converged = (
                iter_count == self.max_iter or 
                (delta_score > 0 and delta_score < self.tol)
            )
            elapsed_time = time.time() - start
            if self.verbose:
                self._print_status(iter_count,pop,fitness,delta_score,elapsed_time)
        self.n_iter = iter_count
        self.coclusters = self.pop[np.nanmean(fitness,axis=(1,2)).argmin() if self.minimize else np.nanmean(fitness,axis=(1,2)).argmax()]
        self.n_row_clusters, self.n_col_clusters  = np.unique(pop[-1][0]).size, np.unique(pop[-1][1]).size
        self.models = self._initialize_models(fit_mask,self.coclusters)
        self.models,_ = self._update_models(data,fit_mask,self.coclusters,self.models)

        
    def predict(self,matrix,row_features,col_features,pred_mask=None):
        data = (matrix,row_features,col_features)
        if pred_mask is None:
            pred_mask = np.isnan(matrix)
        pred_matrix = matrix.copy()
        results = self._predict_coclusters(data,pred_mask,self.coclusters,self.models)
        for i in range(self.n_row_clusters):
            for j in range(self.n_col_clusters):
                cocluster_mask = self._get_bool_mask(self.coclusters,i,j)
                pred_matrix[cocluster_mask&pred_mask] = results[i][j]

        return pred_matrix

    def score(self,matrix,row_features,col_features,pred_mask=None):
        data = (matrix,row_features,col_features)
        if pred_mask is None:
            pred_mask = np.invert(np.isnan(matrix))
        scores = self._score_coclusters(data,pred_mask,self.coclusters,self.models)
        score = np.mean(np.array(scores))

        return score


  