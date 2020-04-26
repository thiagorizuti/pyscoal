import numpy as np
import time
from sklearn.base import clone
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
        self.random_state=42
        self.n_jobs=1

    def _random_init(self,mask,n_row_clusters,n_col_clusters):
        n_rows, n_cols = mask.shape
        np.random.seed(self.random_state) 
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
        else:
            row_clusters, col_clusters = self._random_init(mask,n_row_clusters,n_col_clusters)
        coclusters = (row_clusters, col_clusters)

        return coclusters

    def _initialize_models(self,mask,n_row_clusters,n_col_clusters):
        models = [[clone(self.estimator) for j in range(n_col_clusters)] 
                            for i in range(n_row_clusters)]

        return models

    def _check_coclusters(self,mask,coclusters):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        print(n_row_clusters, n_col_clusters)
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

    def _get_X_y(self,data,mask,rows,cols):
        matrix, row_features,col_features = data
        mask = mask[np.ix_(rows,cols)].ravel()
        X = np.hstack([np.repeat(row_features[rows], col_features[cols].shape[0], axis=0),
            np.tile(col_features[cols], (row_features[rows].shape[0], 1))])
        X = X[mask]
        y = matrix[np.ix_(rows,cols)].ravel()
        y = y[mask]

        return X, y

    def _fit(self,data,mask,rows,cols,model):
        X, y = self._get_X_y(data,mask,rows,cols)
        model.fit(X,y)

        return model 
    
    def _predict(self,data,mask,rows,cols,model):
        X,_ = self._get_X_y(data,mask,rows,cols)
        y_pred = model.predict(X)

        return y_pred 
    
    def _score(self,data,mask,rows,cols,model):
        X, y = self._get_X_y(data,mask,rows,cols)
        y_pred = model.predict(X)
        score = self.scoring(y,y_pred)

        return score 
 
    def _compute_parallel(self,data,mask,coclusters,models,function):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size

        parallel_results = Parallel(n_jobs=self.n_jobs)(delayed(function)
            (data,mask,*self._get_rows_cols(coclusters,i,j),models[i][j]) 
            for i in range(n_row_clusters) for j in range(n_col_clusters))

        results =  [[parallel_results[i*n_col_clusters+j] 
            for j in range(n_col_clusters)] for i in range(n_row_clusters)]

        return results

    def _fit_coclusters(self,data,mask,coclusters,models):        
        models = self._compute_parallel(data,mask,coclusters,models,self._fit)
    
        return models 

    def _score_coclusters(self,data,mask,coclusters,models):
        scores = self._compute_parallel(data,mask,coclusters,models,self._score)

        return scores
    
    def _predict_coclusters(self,data,mask,coclusters,models):
        predictions = self._compute_parallel(data,mask,coclusters,models,self._predict)

        return predictions

    
    def _score_row_clusters(self,data,mask,coclusters,models):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        n_rows, _ = mask.shape
        matrix, _, _ = data

        scores = np.zeros((n_rows,n_row_clusters))
        for i in range(n_row_clusters):
            for j in range(n_col_clusters):
                rows,cols = self._get_rows_cols(coclusters,None,j) 
                true = matrix[np.ix_(rows,cols)]
                pred = np.copy(true)
                pred[mask[np.ix_(rows,cols)]] = self._predict(data,mask,rows,cols,models[i][j])
                for r in range(n_rows):
                    y_true = true[r,:]
                    y_pred = pred[r,:]
                    if np.isnan(y_true).all():
                        scores[r,i] += 0
                    else:
                        scores[r,i] += self.scoring(y_true[~np.isnan(y_true)],y_pred[~np.isnan(y_pred)])

        return scores

    def _score_col_clusters(self,data,mask,coclusters,models):
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        _, n_cols = mask.shape
        matrix, _, _ = data

        scores = np.zeros((n_cols,n_col_clusters))
        for i in range(n_row_clusters):
            for j in range(n_col_clusters):
                rows,cols = self._get_rows_cols(coclusters,i,None)
                true = matrix[np.ix_(rows,cols)]
                pred = np.copy(true)
                pred[mask[np.ix_(rows,cols)]] = self._predict(data,mask,rows,cols,models[i][j])
                for c in range(n_cols):
                    y_true = true[:,c]
                    y_pred = pred[:,c]
                    if np.isnan(y_true).all():
                        scores[c,j] += 0
                    else:
                        scores[c,j] += self.scoring(y_true[~np.isnan(y_true)],y_pred[~np.isnan(y_pred)])

        return scores

    def _update_row_clusters(self,data,mask,coclusters,models):
        scores = self._score_row_clusters(data,mask,coclusters,models)
        new_row_clusters  = np.argmin(scores,axis=1) if self.minimize else np.argmax(scores,axis=1)

        return new_row_clusters
    
    def _update_col_clusters(self,data,mask,coclusters,models):
        scores = self._score_col_clusters(data,mask,coclusters,models)
        new_col_clusters  = np.argmin(scores,axis=1) if self.minimize else np.argmax(scores,axis=1)

        return new_col_clusters

    def _update_coclusters(self,data,mask,coclusters,models):
        new_row_clusters = self._update_row_clusters(data,mask,coclusters,models)
        new_col_clusters = self._update_col_clusters(data,mask,coclusters,models)

        return new_row_clusters,new_col_clusters

    
    # def predict(self,matrix,row_features,col_features,pred_mask=None):
        
    #     if pred_mask is None:
    #         pred_mask = np.isnan(matrix)

    #     prediction = self._predict_models(matrix,row_features,col_features,pred_mask)

    #     return prediction

    
    # def score(self,matrix,row_features,col_features,score_mask=None):
        
    #      if score_mask is None:
    #          score_mask = np.invert(np.isnan(matrix))

    #      scores = np.mean(self._score_models(matrix,row_features,col_features,score_mask))
      
    #      return scores

class SCOAL(BaseScoal):
    
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

    def _print_status(self,iter_count,score,delta_score,rows_changed,cols_changed,elapsed_time):
        if iter_count==0:
            print('|'.join(x.ljust(15) for x in [
                    'iteration',' score','delta score','rows changed', 'columns changed', 'elapsed time (s)']))

        print('|'.join(x.ljust(15) for x in ['%i' % iter_count,'%.3f' % score,'%.3f' % delta_score,'%i' % rows_changed,'%i'  % cols_changed,'%i' % elapsed_time]))

    def fit(self,matrix,row_features,col_features,fit_mask=None):
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

        self.models = self._initialize_models(fit_mask,self.n_row_clusters,self.n_col_clusters)
        self.coclusters = self._initialize_coclusters(fit_mask,self.n_row_clusters,self.n_col_clusters)
        print(self._check_coclusters(fit_mask,self.coclusters).all())
        self.models = self._fit_coclusters(data,fit_mask,self.coclusters,self.models)
        scores = self._score_coclusters(data,fit_mask,self.coclusters,self.models)
        score = np.mean(scores)
        
        if self.verbose:
            self._print_status(iter_count,score,delta_score,rows_changed,cols_changed,elapsed_time)

        while not converged:
            iter_count += 1

            new_row_clusters, new_col_clusters = self._update_coclusters(data,fit_mask,self.coclusters,self.models)     
            rows_changed = np.sum(new_row_clusters!=self.coclusters[0])
            cols_changed = np.sum(new_col_clusters!=self.coclusters[1])
            self.coclusters = (np.copy(new_row_clusters), np.copy(new_col_clusters))
            print(self._check_coclusters(fit_mask,self.coclusters).all())
            delta_score = score
            self.models = self._fit_coclusters(data,fit_mask,self.coclusters,self.models)
            scores = self._score_coclusters(data,fit_mask,self.coclusters,self.models)
            score = np.mean(scores)            
            delta_score -= score

            converged = (
                iter_count >= self.max_iter or
                (delta_score > 0 and delta_score < self.tol) or
                (rows_changed==0 and cols_changed==0)
            )   

            elapsed_time = time.time() - start

            if self.verbose:
                self._print_status(iter_count,score,delta_score,rows_changed,cols_changed,elapsed_time)

class EvolutiveScoal(BaseScoal):
    
    def __init__(self,
                max_row_clusters=10,
                max_col_clusters=10,
                estimator=LinearRegression(),
                scoring=mean_squared_error,
                minimize=True,
                pop_size=20,
                max_iter=np.nan,
                tol = 1e-3,
                init='random',
                random_state=42,
                n_jobs=1,
                verbose=False):

        self.max_row_clusters=max_row_clusters
        self.max_col_clusters=max_col_clusters
        self.estimator=estimator
        self.scoring=mean_squared_error
        self.minimize=minimize
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.tol=tol
        self.init=init
        self.random_state=random_state
        self.n_jobs=n_jobs
        self.verbose=verbose

    # def _local_search(self,data,fit_mask,pop):
    #     new_pop = []
    #     for ind in pop:
    #         models = [[clone(self.estimator) for i in range(self.n_row_clusters)] 
    #                         for j in range(self.n_col_clusters) ]
    #         estimators = self._fit_coclusters(data,fit_mask,ind)
    #         new_row_clusters,new_col_clusters = self._update_coclusters(data,fit_mask,ind,estimators)
    #         new_pop.append((new_row_clusters,new_col_clusters))
    #     return new_pop

    def _delete_cluster(self,clusters,cluster):
        n_clusters = np.unique(clusters).size
        if n_clusters > 1:
            clusters[clusters==cluster] = -1
            clusters[clusters>cluster] -= 1
            clusters[clusters==-1] = np.random.choice(np.arange(n_clusters-1),(clusters==-1).sum())

        return clusters

    def _mutation(self,pop,fitness):
        new_pop = []
        for i, ind in enumerate(pop):
            row_clusters,col_clusters = ind
            n_row_clusters, n_col_clusters = np.unique(row_clusters).size, np.unique(col_clusters).size
            probs = np.concatenate((np.nansum(fitness[i],axis=0),
                                    np.nansum(fitness[i],axis=1)),axis=0)
            probs = probs/probs.sum()
            choice = np.random.choice(np.arange(probs.size),p=probs)
            if choice < n_row_clusters:
                dim='row'
                cluster=choice
            else:
                dim='col'
                cluster=choice-n_row_clusters
            if dim == 'row':
                row_clusters=self._delete_cluster(row_clusters,cluster)
            if dim == 'col':
                col_clusters=self._delete_cluster(col_clusters,cluster)
            new_pop.append((row_clusters,col_clusters))
        return new_pop
   
    def _replacement(self,pop,fitness,new_pop,new_fitness):

        fitness = np.concatenate((fitness,new_fitness),axis=0)
        probs = np.nanmean(fitness,axis=(1,2))
        best = np.argmax(probs)
        probs = probs/probs.sum()
        probs[best] = 1.
        selected = np.random.choice(np.arange(probs.size),self.pop_size-1,replace=False,p=probs)
        pop = pop + new_pop
        pop = pop[selected]

        return pop 
            
    def _evaluate_fitness(self,data,fit_mask,test_mask,pop):
        fitness = np.zeros((self.pop_size,self.max_row_clusters,self.max_col_clusters))*np.nan
        for i,ind in enumerate(pop):
            models = self._fit_coclusters(data,fit_mask,ind)
            scores = self._score_coclusters(data,test_mask,ind,models)
            n_row_clusters, n_col_clusters = scores.shape
            fitness[i,:n_row_clusters,:n_col_clusters] = 1/scores if self.minimize else scores

        return fitness
       
    def _init_population(self,fit_mask):
        np.random.seed(self.random_state) 
        n_row_clusters = np.random.randint(1,self.max_row_clusters+1,self.pop_size)
        n_col_clusters = np.random.randint(1,self.max_col_clusters+1,self.pop_size)
        pop = [(self._initialize_coclusters(fit_mask,i,j)) for i,j in zip(n_row_clusters,n_col_clusters)]

        return pop

    def _check_population(self,mask,pop):
        valid = np.zeros(self.pop_size).astype(bool)
        for i,ind in enumerate(pop):
            valid[i] = (self._check_coclusters(mask,ind)).all()

        return valid

    def fit(self,matrix,row_features,col_features,fit_mask=None):
        data = matrix, row_features, col_features
        if fit_mask is None:
            fit_mask = np.invert(np.isnan(matrix)) 
        test = np.random.choice(np.arange(fit_mask.sum()), int(fit_mask.sum()*0.3),replace=False)
        rows,cols = np.where(fit_mask)
        test_mask = np.zeros(fit_mask.shape).astype(bool)
        test_mask[rows[test],cols[test]] = True
        fit_mask[rows[test],cols[test]] = False

        iter_count = 0
        converged = False

        pop = self._init_population(fit_mask)     
        print((self._check_population(fit_mask,pop)).all())  
        # pop = self._local_search(data,fit_mask,pop)
        self.pop = pop 
        print((self._check_population(fit_mask,pop)).all())  
        fitness = self._evaluate_fitness(data,fit_mask,test_mask,pop)
        #new_pop = self._mutation(pop,fitness)
        print((self._check_population(fit_mask,pop)).all())  
        #new_fitness = self._evaluate_fitness(matrix,row_features,col_features,fit_mask,test_mask,pop)
        #pop = self._replacement(pop,fitness,new_pop,new_fitness)
        print((self._check_population(fit_mask,pop)).all())  


        # while(not converged):
        #     print(iter_count, np.nanmean(fitness,axis=(1,2)).min())
        #     iter_count+=1
        #     converged = iter_count > self.max_iter

  