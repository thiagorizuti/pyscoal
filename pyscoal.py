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
        self.init='random'
        self.random_state=42,
        self.n_jobs=1,

    def _random_init(self,n_rows,n_cols,n_row_clusters,n_col_clusters):
        np.random.seed(self.random_state) 
        row_clusters = np.random.choice(np.arange(n_row_clusters),n_rows)
        col_clusters = np.random.choice(np.arange(n_col_clusters),n_cols)

        return row_clusters, col_clusters 
        
    def _smart_init(self,mask,n_rows,n_cols,n_row_clusters,n_col_clusters):

        return None,None

    def _initialize_coclusters(self,mask,n_row_clusters,n_col_clusters):
        n_rows, n_cols = mask.shape

        if self.init=='smart':
            row_clusters, col_clusters = self._smart_init(mask,n_rows,n_cols,n_row_clusters,n_col_clusters)
        else:
            row_clusters, col_clusters = self._random_init(n_rows,n_cols,n_row_clusters,n_col_clusters)

        return row_clusters, col_clusters

    def _check_coclusters(self,mask,row_clusters,col_clusters):
        n_row_clusters,n_col_clusters = np.unique(row_clusters).size, np.unique(col_clusters).size
        
        valid = np.zeros((n_row_clusters,n_col_clusters)).astype(bool)
        for i in range(n_row_clusters):
            for j in range(n_col_clusters):
                rows,cols = self._get_rows_cols(row_clusters,col_clusters,i,j)
                valid[i,j] = mask[np.ix_(rows,cols)].any()
    
        return valid

    def _get_rows_cols(self,row_clusters,col_clusters,row_cluster,col_cluster):
        if row_cluster is None:
            rows_mask = np.ones(row_clusters.shape).astype(bool)
        else:
            rows_mask = row_clusters==row_cluster
        if col_cluster is None:
            cols_mask = np.ones(col_clusters.shape).astype(bool)
        else:
            cols_mask = col_clusters==col_cluster
        rows = np.argwhere(rows_mask).ravel()
        cols = np.argwhere(cols_mask).ravel()

        return rows,cols

    def _get_X(self,row_features,col_features,mask,rows,cols):
        mask = mask[np.ix_(rows,cols)].ravel()
        X = np.hstack([np.repeat(row_features[rows], col_features[cols].shape[0], axis=0),
            np.tile(col_features[cols], (row_features[rows].shape[0], 1))])
        X = X[mask]

        return X

    def _get_y(self,matrix,mask,rows,cols):
        mask = mask[np.ix_(rows,cols)].ravel()
        y = matrix[np.ix_(rows,cols)].ravel()
        y = y[mask]
       
        return y

    def _fit(self,matrix,row_features,col_features,mask,rows,cols):
        X = self._get_X(row_features,col_features,mask,rows,cols)
        y = self._get_y(matrix,mask,rows,cols)
        estimator = clone(self.estimator)
        estimator.fit(X,y)

        return estimator 
    
    def _predict(self,row_features,col_features,mask,rows,cols,estimator):
        X = self._get_X(row_features,col_features,mask,rows,cols)
        y_pred = estimator.predict(X)

        return y_pred 
    
    def _score(self,matrix,row_features,col_features,mask,rows,cols,estimator):
        X = self._get_X(row_features,col_features,mask,rows,cols)
        y = self._get_y(matrix,mask,rows,cols)
        y_pred = estimator.predict(X)
        score = self.scoring(y,y_pred)

        return score 

    def _fit_coclusters(self,matrix,row_features,col_features,mask,row_clusters,col_clusters):
        n_row_clusters,n_col_clusters = np.unique(row_clusters).size, np.unique(col_clusters).size

        estimators = Parallel(n_jobs=self.n_jobs)(delayed(self._fit)
            (matrix,row_features,col_features,mask,*self._get_rows_cols(row_clusters,col_clusters,i,j)) 
            for i in range(n_row_clusters) for j in range(n_col_clusters))

        estimators =  [[estimators[i*n_col_clusters+j] 
            for j in range(n_col_clusters)] for i in range(n_row_clusters)]

        return estimators 

    def _score_coclusters(self,matrix,row_features,col_features,mask,row_clusters,col_clusters,estimators):
        n_row_clusters,n_col_clusters = np.unique(row_clusters).size, np.unique(col_clusters).size

        scores = Parallel(n_jobs=self.n_jobs)(delayed(self._score)
            (matrix,row_features,col_features,mask,*self._get_rows_cols(row_clusters,col_clusters,i,j),estimators[i][j])  
            for i in range(n_row_clusters) for j in range(n_col_clusters))

        scores = np.array([[scores[i*n_col_clusters+j] 
            for j in range(n_col_clusters)] for i in range(n_row_clusters)])

        return scores
    
    # def _predict_models(self,matrix,row_features,col_features,fit_mask):
    #     predictions = Parallel(n_jobs=self.n_jobs)(delayed(self._predict_model)
    #         (self._get_X(row_features,col_features,i,j),self.estimators[i][j]) 
    #         for i in range(self.n_row_clusters) for j in range(self.n_col_clusters))

    #     predictions = [[predictions[i*self.n_col_clusters+j] 
    #         for j in range(self.n_col_clusters)] for i in range(self.n_row_clusters)]

    #     return predictions

    def _score_row_clusters(self,matrix,row_features,col_features,mask,row_clusters,col_clusters,estimators):
        n_rows, _ = matrix.shape
        n_row_clusters,n_col_clusters = np.unique(row_clusters).size, np.unique(col_clusters).size

        predictions = Parallel(n_jobs=self.n_jobs)(delayed(self._predict)
            (row_features,col_features,mask,*self._get_rows_cols(row_clusters,col_clusters,None,j),estimators[i][j])  
            for i in range(n_row_clusters) for j in range(n_col_clusters))

        scores = np.zeros((n_rows,n_row_clusters))
        for i in range(n_row_clusters):
            for j in range(n_col_clusters):
                rows,cols = self._get_rows_cols(row_clusters,col_clusters,None,j)
                true = matrix[np.ix_(rows,cols)]
                pred = np.copy(true)
                pred[mask[np.ix_(rows,cols)]] = predictions[i*n_col_clusters+j]
                for r in range(n_rows):
                    y_true = true[r,:]
                    y_pred = pred[r,:]
                    if np.isnan(y_true).all():
                        scores[r,i] += 0
                    else:
                        scores[r,i] += self.scoring(y_true[~np.isnan(y_true)],y_pred[~np.isnan(y_pred)])
            scores = scores/n_col_clusters
        return scores

    def _score_col_clusters(self,matrix,row_features,col_features,mask,row_clusters,col_clusters,estimators):
        _, n_cols = matrix.shape
        n_row_clusters,n_col_clusters = np.unique(row_clusters).size, np.unique(col_clusters).size

        predictions = Parallel(n_jobs=self.n_jobs)(delayed(self._predict)
            (row_features,col_features,mask,*self._get_rows_cols(row_clusters,col_clusters,i,None),estimators[i][j])  
            for i in range(n_row_clusters) for j in range(n_col_clusters))

        scores = np.zeros((n_cols,n_col_clusters))
        for i in range(n_row_clusters):
            for j in range(n_col_clusters):
                rows,cols = self._get_rows_cols(row_clusters,col_clusters,i,None)
                true = matrix[np.ix_(rows,cols)]
                pred = np.copy(true)
                pred[mask[np.ix_(rows,cols)]] = predictions[i*n_col_clusters+j]
                for c in range(n_cols):
                    y_true = true[:,c]
                    y_pred = pred[:,c]
                    if np.isnan(y_true).all():
                        scores[c,j] += 0
                    else:
                        scores[c,j] += self.scoring(y_true[~np.isnan(y_true)],y_pred[~np.isnan(y_pred)])
            scores = scores/n_row_clusters
        return scores

    def _update_row_clusters(self,matrix,row_features,col_features,mask,row_clusters,col_clusters,estimators):
        scores = self._score_row_clusters(matrix,row_features,col_features,mask,row_clusters,col_clusters,estimators)
        row_clusters  = np.argmin(scores,axis=1) if self.minimize else np.argmax(scores,axis=1)
        return row_clusters
    
    def _update_col_clusters(self,matrix,row_features,col_features,mask,row_clusters,col_clusters,estimators):
        scores = self._score_col_clusters(matrix,row_features,col_features,mask,row_clusters,col_clusters,estimators)
        col_clusters  = np.argmin(scores,axis=1) if self.minimize else np.argmax(scores,axis=1)
        return col_clusters

    def _update_coclusters(self,matrix,row_features,col_features,mask,row_clusters,col_clusters,estimators):
        new_row_clusters = self._update_row_clusters(matrix,row_features,col_features,mask,row_clusters,col_clusters,estimators)
        new_col_clusters = self._update_col_clusters(matrix,row_features,col_features,mask,new_row_clusters,col_clusters,estimators)
        return new_row_clusters,new_col_clusters

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
        
        if fit_mask is None:
            fit_mask = np.invert(np.isnan(matrix))         
        self.fit_mask = fit_mask
        self.n_rows, self.n_cols = matrix.shape

        iter_count=0 
        elapsed_time = 0
        rows_changed = 0
        cols_changed = 0
        score = np.nan
        delta_score=np.nan
        converged = False
        start = time.time()

        self.row_clusters,self.col_clusters = self._initialize_coclusters(self.fit_mask,self.n_row_clusters,self.n_col_clusters)
        self.estimators = self._fit_coclusters(matrix,row_features,col_features,self.fit_mask,self.row_clusters,self.col_clusters)
        score = np.mean(self._score_coclusters(matrix,row_features,col_features,self.fit_mask,self.row_clusters,self.col_clusters,self.estimators))
        
        if self.verbose:
            self._print_status(iter_count,score,delta_score,rows_changed,cols_changed,elapsed_time)

        while not converged:
            iter_count += 1

            new_row_clusters, new_col_clusters = self._update_coclusters(matrix,row_features,col_features,self.fit_mask,self.row_clusters,self.col_clusters,self.estimators)     
            rows_changed = np.sum(new_row_clusters!=self.row_clusters)
            cols_changed = np.sum(new_col_clusters!=self.col_clusters)
            self.row_clusters = np.copy(new_row_clusters)
            self.col_clusters = np.copy(new_col_clusters)
        
            delta_score = score
            self.estimators=self._fit_coclusters(matrix,row_features,col_features,self.fit_mask,self.row_clusters,self.col_clusters)
            score = np.mean(self._score_coclusters(matrix,row_features,col_features,self.fit_mask,self.row_clusters,self.col_clusters,self.estimators))
            delta_score -= score

            converged = (
                iter_count >= self.max_iter or
                (delta_score > 0 and delta_score < self.tol) or
                (rows_changed==0 and cols_changed==0)
            )   

            elapsed_time = time.time() - start

            if self.verbose:
                self._print_status(iter_count,score,delta_score,rows_changed,cols_changed,elapsed_time)

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

class EvolutiveScoal(BaseScoal):
    
    def __init__(self,
                max_row_clusters=10,
                max_col_clusters=10,
                estimator=LinearRegression(),
                scoring=mean_squared_error,
                minimize=True,
                pop_size=20,
                cx_rate=0.7,
                mut_rate=0.1,
                elitism=0.05,
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
        self.cx_rate = cx_rate
        self.mut_rate = mut_rate
        self.elitism = elitism
        self.max_iter = max_iter
        self.tol=tol
        self.init=init
        self.random_state=random_state
        self.n_jobs=n_jobs
        self.verbose=verbose

    def _local_search(self,matrix,row_features,col_features,test_mask,fit_mask,ind,fitness):
        row_clusters,col_clusters,n_row_clusters,n_col_clusters = ind

        estimators = self._fit_coclusters(matrix,row_features,col_features,fit_mask,row_clusters,col_clusters)
        new_row_clusters,new_col_clusters = self._update_coclusters(matrix,row_features,col_features,test_mask,row_clusters,col_clusters,estimators)
       
        return (new_row_clusters,new_col_clusters,n_row_clusters,n_col_clusters)

    def _replacement(self, new_pop):
        if self.elitism > 0:
           pass
    
    def _delete_cluster(self,clusters,cluster,n_clusters):
        clusters[clusters==cluster] = -1
        clusters[clusters>cluster] -= 1
        clusters[clusters==-1] = np.random.choice(np.arange(n_clusters),(clusters==-1).sum())

        return clusters

    def _mutation(self,ind,fitness):
        row_clusters,col_clusters,n_row_clusters,n_col_clusters = ind

        if np.random.random() < self.mut_rate:
            probs = np.nan_to_num(fitness.ravel())
            probs = probs/probs.sum()
            choice = np.random.choice(np.arange(probs.size),p=probs)
            row_cluster,col_cluster = np.unravel_index(choice,fitness.shape)
            dim = np.random.randint(2) 
            if dim == 0 and n_row_clusters > 1 :
                n_row_clusters -= 1
                row_clusters=self._delete_cluster(row_clusters,row_cluster,n_row_clusters)
            if dim == 1 and n_col_clusters > 1:
                n_col_clusters -= 1
                col_clusters=self._delete_cluster(col_clusters,col_cluster,n_col_clusters)

        return (row_clusters,col_clusters,n_row_clusters,n_col_clusters)

            
    def _crossover(self,ind1,ind2):
        if np.random.random() < self.cx_rate:
            pass
                 
    def _reproduction(self,selected):
        new_pop = [self._mutation(self.pop[i],self.fitness[i]) for i in selected]
        return new_pop
        
    def _selection(self):
        probs = np.nanmean(self.fitness,axis=(1,2))
        probs = probs/probs.sum()
        selected = np.random.choice(np.arange(self.pop_size),self.pop_size,replace=True,p=probs)
        
        return selected
    
    def _evaluate_fitness(self,matrix,row_features,col_features,fit_mask,test_mask):
        fitness = np.zeros((self.pop_size,self.max_row_clusters,self.max_col_clusters))*np.nan
        for i,ind in enumerate(self.pop):
            row_clusters,col_clusters,n_row_clusters,n_col_clusters = ind
            #n_row_clusters = np.unique(row_clusters).size,
            #n_col_clusters = np.unique(col_clusters).size
            estimators = self._fit_coclusters(matrix,row_features,col_features,fit_mask,row_clusters,col_clusters)
            scores = self._score_coclusters(matrix,row_features,col_features,test_mask,row_clusters,col_clusters,estimators)
            fitness[i,:scores.shape[0],:scores.shape[1]] = scores
        return fitness
       
    def _init_pop(self,fit_mask):
        np.random.seed(self.random_state) 
        n_row_clusters = np.random.randint(1,self.max_row_clusters+1,self.pop_size)
        n_col_clusters = np.random.randint(1,self.max_col_clusters+1,self.pop_size)
        pop = [(*self._initialize_coclusters(fit_mask,i,j),i,j) for i,j in zip(n_row_clusters,n_col_clusters)]

        return pop
        
            
    def fit(self,matrix,row_features,col_features,fit_mask=None):
        if fit_mask is None:
            fit_mask = np.invert(np.isnan(matrix)) 
        test = np.random.choice(np.arange(fit_mask.sum()), int(fit_mask.sum()*0.3),replace=False)
        rows,cols = np.where(fit_mask)
        test_mask = np.zeros(fit_mask.shape).astype(bool)
        test_mask[rows[test],cols[test]] = True
        fit_mask[rows[test],cols[test]] = False

        iter_count = 0
        converged = False

        self.pop = self._init_pop(fit_mask)
        self.fitness = self._evaluate_fitness(matrix,row_features,col_features,fit_mask,test_mask)
       

        while(not converged):
          
            print(iter_count, np.nanmean(self.fitness,axis=(1,2)).min())
            selected = self._selection()
            self.pop = self._reproduction(selected)
            self.fitness = self._evaluate_fitness(matrix,row_features,col_features,fit_mask,test_mask)
            iter_count+=1
            converged = iter_count > self.max_iter
            
  