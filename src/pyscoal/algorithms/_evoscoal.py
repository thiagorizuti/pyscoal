import numpy as np
import time
from sklearn.base import is_regressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from joblib import Parallel, delayed, Memory
from copy import deepcopy
from ._scoal import SCOAL

## implementar alocao scoal-like,aleatorio e knn 

class EvoSCOAL(SCOAL):
    
    def __init__(self,
                max_row_clusters=5,
                max_col_clusters=5,
                pop_size=5,
                max_gen=50,
                gen_tol=10,
                tol=0.01,
                estimator=LinearRegression(),
                validation_size=0.2,
                mutation_strength='max',
                fitness_function = 'SSE',
                random_state=42,
                n_jobs=(1,1),
                cache=False,
                matrix='sparse',
                verbose=False):

        self.max_row_clusters=max_row_clusters
        self.max_col_clusters=max_col_clusters
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.gen_tol = gen_tol
        self.tol = tol
        self.estimator=estimator
        self.validation_size=validation_size
        self.fitness_function = fitness_function
        self.random_state=random_state
        self.n_jobs=n_jobs
        self.cache=cache
        self.verbose=verbose
        self.matrix=matrix
        self.is_regressor = is_regressor(estimator)
        self.mutation_strength = max_row_clusters*max_col_clusters if mutation_strength == 'max' else mutation_strength


    def _baesian_information_criterion(self,data,coclusters,models,n_jobs):
        log_likelihood = self._compute_clusterwise(data,coclusters,models,self._log_likelihood,n_jobs)
        row_clusters, col_clusters = coclusters
        n_row_clusters, n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        n_params = self.n_rows + self.n_cols + (n_row_clusters*n_col_clusters)*(self.n_row_features+self.n_col_features)
        bic = n_params*np.log(self.n_values) - 2*np.sum(log_likelihood)
        

        return bic

    def _log_likelihood(self,data,coclusters,models,row_cluster,col_cluster):
        rows,cols = self._get_rows_cols(coclusters,row_cluster,col_cluster)
        X, y = self._get_X_y(data,rows,cols)
        model = models[row_cluster][col_cluster]
        log_likelihood = 0
        if y.size > 0:
            y_pred = model.predict(X) if self.is_regressor else model.predict_proba(X)[:,1]
            log_likelihood = np.nansum(norm.logpdf(y,scale=y_pred,loc=mean_squared_error(y,y_pred)))

        return log_likelihood 
        
    def _heuristic_init(self,train_data,n_row_clusters,n_col_clusters):
        matrix, _, _ = train_data
        if self.matrix=='dense':
            mask = np.invert(np.isnan(matrix))
        else:
            rows, cols, values = matrix[:,0].astype(int), matrix[:,1].astype(int), matrix[:,2]
            mask = np.zeros((self.n_rows, self.n_cols))
            mask[rows,cols] = values
            mask = np.invert(np.isnan(mask))
        row_clusters = np.zeros(self.n_rows)*np.nan
        col_clusters = np.zeros(self.n_cols)*np.nan
        row_clusters_aux = np.zeros((n_row_clusters,self.n_cols))
        for row in np.random.choice(np.arange(self.n_rows),self.n_rows,replace=False):
            fills = (mask[row,:]>row_clusters_aux).sum(axis=1)
            row_cluster = np.random.choice(np.where(fills == fills.max())[0])
            row_clusters_aux[row_cluster]=np.logical_or(row_clusters_aux[row_cluster],mask[row,:])
            row_clusters[row] = row_cluster
        col_clusters_aux = np.zeros((n_col_clusters,n_row_clusters))
        for col in np.random.choice(np.arange(self.n_cols),self.n_cols,replace=False):
            fills = (row_clusters_aux[:,col]>col_clusters_aux).sum(axis=1)
            col_cluster = np.random.choice(np.where(fills == fills.max())[0])
            col_clusters_aux[col_cluster]=np.logical_or(col_clusters_aux[col_cluster],row_clusters_aux[:,col])
            col_clusters[col] = col_cluster
        row_clusters = row_clusters.astype(int)
        col_clusters = col_clusters.astype(int)

        return row_clusters, col_clusters
    
    def _initialize_ind(self,train_data):
        n_row_clusters = np.random.randint(1,self.max_row_clusters+1)
        n_col_clusters = np.random.randint(1,self.max_col_clusters+1)
        row_clusters, col_clusters = self._heuristic_init(train_data,n_row_clusters,n_col_clusters)
        coclusters = (row_clusters, col_clusters)

        return coclusters
                
    def _initialize_pop(self,train_data):
        population = [self._initialize_ind(train_data) for i in range(self.pop_size)]
        fitness = np.zeros((self.pop_size,2,self.max_row_clusters,self.max_col_clusters))*np.nan
        scores = np.zeros(self.pop_size)*np.nan

        return population, fitness, scores
    
    def _compute_individualwise(self,train_data,valid_data,population,fitness,scores,function,n_jobs):
        results = Parallel(n_jobs=n_jobs[0])(delayed(function)
            (train_data,valid_data,population[i],fitness[i],scores[i],n_jobs[1]) 
            for i in range(self.pop_size))

        return results
                   
    def _check_pop(self,train_data,valid_data,population,fitness,scores,n_jobs):
        results = self._compute_individualwise(train_data,valid_data,population,fitness,scores,self._check_individual,n_jobs)
        checked = np.array(results)

        return checked

    def _check_ind(self,train_data,valid_data,individual,fitness,scores,n_jobs):
        results = self._check_coclusters(train_data,individual,None,n_jobs=1)
        checked = np.all(results)

        return checked

    def _evaluate_fitness_pop(self,train_data,valid_data,population,fitness,scores,n_jobs):
            results = self._compute_individualwise(train_data,valid_data,population,fitness,scores,self._evaluate_fitness_ind,n_jobs)
            fitness = np.array([result[0] for result in results])
            scores = np.array([result[1] for result in results])

            return fitness, scores

    def _evaluate_fitness_ind(self,train_data,valid_data,individual,fitness,scores,n_jobs):
        fitness = np.nan
        scores = np.zeros((2,self.max_row_clusters,self.max_col_clusters))*np.nan
        check = self._check_coclusters(train_data,individual,None,n_jobs=1)
        if  np.all(check):
            row_clusters,col_clusters = individual
            n_row_clusters, n_col_clusters = np.unique(row_clusters).size, np.unique(col_clusters).size
            models = self._initialize_models(individual)
            models, train_scores = self._update_models(train_data,individual,models,n_jobs)
            if self.fitness_function == 'BIC':
                scores[0,:n_row_clusters,:n_col_clusters] = train_scores
                scores[1,:n_row_clusters,:n_col_clusters] = check
                fitness = self._baesian_information_criterion(train_data,individual,models,n_jobs) 
            else:
                scores[0,:n_row_clusters,:n_col_clusters] = self._score_coclusters(valid_data,individual,models,n_jobs)
                scores[1,:n_row_clusters,:n_col_clusters] = self._check_coclusters(valid_data,individual,models,n_jobs)
                fitness = np.nansum(scores[0],axis=(0,1))/np.nansum(scores[1],axis=(0,1))
             
        return fitness, scores

    def _local_search_pop(self,train_data,valid_data,population,fitness,scores,n_jobs):
        results = self._compute_individualwise(train_data,valid_data,population,fitness,scores,self._local_search_ind,n_jobs)
        new_population = results

        return new_population  

    def _local_search_ind(self,train_data,valid_data,individual,fitness,scores,n_jobs):
        if not np.isnan(fitness):
            models = self._initialize_models(individual)
            models,_ = self._update_models(train_data,individual,models,n_jobs)
            individual = self._update_coclusters(train_data,individual,models,n_jobs)

        return individual 

    def _muatation_pop(self,train_data,valid_data,population,fitness,scores,n_jobs):
            results = self._compute_individualwise(train_data,valid_data,population,fitness,scores,self._mutation_ind,n_jobs)
            new_population = results

            return new_population

    def _mutation_ind(self,train_data,valid_data,individual,fitness,scores,n_jobs):
        if not np.isnan(fitness):
            row_clusters,col_clusters = individual
            n_row_clusters, n_col_clusters = np.unique(row_clusters).size, np.unique(col_clusters).size
            probs = np.concatenate((np.nansum(scores[0],axis=1),np.nansum(scores[0],axis=0)),axis=0)
            n_values = np.concatenate((np.nansum(scores[1],axis=1),np.nansum(scores[1],axis=0)),axis=0)
            np.divide(probs,n_values, where=n_values!=0,out=probs)
            probs = np.argsort(np.argsort(probs)) - np.sum(probs==0)
            probs = np.maximum(probs,0)
            probs = probs/probs.sum()
            many_clusters = 1 if self.mutation_strength == 1 else np.random.randint(1,np.minimum(self.mutation_strength,n_row_clusters+n_col_clusters)) 
            choices = np.random.choice(np.arange(probs.size),many_clusters,replace=False,p=probs)
            row_cluster_labels = np.arange(n_row_clusters)
            col_cluster_labels = np.arange(n_col_clusters)
            for choice in choices:  
                if choice < self.max_row_clusters and self.max_row_clusters > 1:
                    row_cluster=row_cluster_labels[choice]
                    split = np.random.random() > (np.unique(row_clusters).size-1)/(self.max_row_clusters-1)
                    if split :
                        row_clusters=self._split_row_cluster(train_data,(row_clusters,col_clusters),row_cluster)
                    else:
                        row_clusters=self._delete_row_cluster(train_data,(row_clusters,col_clusters),row_cluster)
                        row_cluster_labels[row_cluster_labels>row_cluster] -= 1
                elif choice >= self.max_row_clusters and self.max_col_clusters > 1:
                    col_cluster=col_cluster_labels[choice-self.max_row_clusters]
                    split = np.random.random() > (np.unique(col_clusters).size-1)/(self.max_col_clusters-1)
                    if split:
                        col_clusters=self._split_col_cluster(train_data,(row_clusters,col_clusters),col_cluster)
                    else:
                        col_clusters=self._delete_col_cluster(train_data,(row_clusters,col_clusters),col_cluster)
                        col_cluster_labels[col_cluster_labels>col_cluster] -= 1
                else:
                    continue
                individual = row_clusters,col_clusters

        return individual

    def _split_row_cluster(self,train_data,coclusters,row_cluster):
        matrix, _, _ = train_data
        if self.matrix=='dense':
            mask = np.invert(np.isnan(matrix))
        else:
            rows, cols, values = matrix[:,0].astype(int), matrix[:,1].astype(int), matrix[:,2]
            mask = np.zeros((self.n_rows, self.n_cols))
            mask[rows,cols] = values
            mask = np.invert(np.isnan(mask))
        row_clusters,col_clusters = coclusters
        n_row_clusters, n_col_clusters = np.unique(row_clusters).size, np.unique(col_clusters).size
        n_rows, n_cols = mask.shape
        new_row_clusters = np.zeros(n_rows)*np.nan
        col_clusters_aux = np.zeros((n_col_clusters,n_rows))
        row_clusters_aux = np.zeros((n_row_clusters+1,n_col_clusters))
        for col in np.random.choice(np.arange(n_cols),n_cols,replace=False):
                col_cluster = col_clusters[col]
                col_clusters_aux[col_cluster]=np.logical_or(col_clusters_aux[col_cluster],mask[:,col])
        for row in np.random.choice(np.arange(n_rows),n_rows,replace=False):
            if row_clusters[row]==row_cluster:
                fills = (col_clusters_aux[:,row]>row_clusters_aux[[row_cluster,n_row_clusters],:]).sum(axis=1)
                new_row_cluster = np.random.choice(np.where(fills == fills.max())[0])
                new_row_cluster = row_cluster if new_row_cluster == 0 else n_row_clusters
            else:
                new_row_cluster=row_clusters[row]
            row_clusters_aux[new_row_cluster]=np.logical_or(row_clusters_aux[new_row_cluster],col_clusters_aux[:,row])
            new_row_clusters[row] = new_row_cluster
        new_row_clusters = new_row_clusters.astype(int)

        return new_row_clusters

    def _split_col_cluster(self,train_data,coclusters,col_cluster):
        matrix, _, _ = train_data
        if self.matrix=='dense':
            mask = np.invert(np.isnan(matrix))
        else:
            rows, cols, values = matrix[:,0].astype(int), matrix[:,1].astype(int), matrix[:,2]
            mask = np.zeros((self.n_rows, self.n_cols))
            mask[rows,cols] = values
            mask = np.invert(np.isnan(mask))
        row_clusters,col_clusters = coclusters
        n_row_clusters, n_col_clusters = np.unique(row_clusters).size, np.unique(col_clusters).size
        n_rows, n_cols = mask.shape
        new_col_clusters = np.zeros(n_cols)*np.nan
        row_clusters_aux = np.zeros((n_row_clusters,n_cols))
        col_clusters_aux = np.zeros((n_col_clusters+1,n_row_clusters))
        for row in np.random.choice(np.arange(n_rows),n_rows,replace=False):
            row_cluster = row_clusters[row]
            row_clusters_aux[row_cluster]=np.logical_or(row_clusters_aux[row_cluster],mask[row,:])
        for col in np.random.choice(np.arange(n_cols),n_cols,replace=False):        
            if col_clusters[col]==col_cluster:
                fills = (row_clusters_aux[:,col]>col_clusters_aux[[col_cluster,n_col_clusters],:]).sum(axis=1)
                new_col_cluster = np.random.choice(np.where(fills == fills.max())[0])
                new_col_cluster = col_cluster if new_col_cluster == 0 else n_col_clusters
            else:
                new_col_cluster=col_clusters[col]
            col_clusters_aux[new_col_cluster]=np.logical_or(col_clusters_aux[new_col_cluster],row_clusters_aux[:,col])
            new_col_clusters[col] = new_col_cluster
        new_col_clusters = new_col_clusters.astype(int)

        return new_col_clusters

    def _delete_row_cluster(self,train_data,coclusters,row_cluster):
        matrix, _, _ = train_data
        if self.matrix=='dense':
            mask = np.invert(np.isnan(matrix))
        else:
            rows, cols, values = matrix[:,0].astype(int), matrix[:,1].astype(int), matrix[:,2]
            mask = np.zeros((self.n_rows, self.n_cols))
            mask[rows,cols] = values
            mask = np.invert(np.isnan(mask))
        row_clusters,col_clusters = coclusters
        n_row_clusters, n_col_clusters = np.unique(row_clusters).size, np.unique(col_clusters).size
        n_rows, n_cols = mask.shape
        new_row_clusters = np.zeros(n_rows)*np.nan
        col_clusters_aux = np.zeros((n_col_clusters,n_rows))
        row_clusters_aux = np.zeros((n_row_clusters-1,n_col_clusters))
        row_clusters[row_clusters==row_cluster] = -1
        row_clusters[row_clusters>row_cluster] -= 1
        for col in np.random.choice(np.arange(n_cols),n_cols,replace=False):
            col_cluster = col_clusters[col]
            col_clusters_aux[col_cluster]=np.logical_or(col_clusters_aux[col_cluster],mask[:,col])  
        for row in np.random.choice(np.arange(n_rows),n_rows,replace=False):
            if row_clusters[row]==-1:
                fills = (col_clusters_aux[:,row]>row_clusters_aux).sum(axis=1)
                new_row_cluster = np.random.choice(np.where(fills == fills.max())[0])
            else:
                new_row_cluster=row_clusters[row]
            row_clusters_aux[new_row_cluster]=np.logical_or(row_clusters_aux[new_row_cluster],col_clusters_aux[:,row])
            new_row_clusters[row] = new_row_cluster
        new_row_clusters = new_row_clusters.astype(int)

        return new_row_clusters

    def _delete_col_cluster(self,train_data,coclusters,col_cluster):
        matrix, _, _ = train_data
        if self.matrix=='dense':
            mask = np.invert(np.isnan(matrix))
        else:
            rows, cols, values = matrix[:,0].astype(int), matrix[:,1].astype(int), matrix[:,2]
            mask = np.zeros((self.n_rows, self.n_cols))
            mask[rows,cols] = values
            mask = np.invert(np.isnan(mask))
        row_clusters,col_clusters = coclusters
        n_row_clusters, n_col_clusters = np.unique(row_clusters).size, np.unique(col_clusters).size
        n_rows, n_cols = mask.shape
        new_col_clusters = np.zeros(n_cols)*np.nan
        row_clusters_aux = np.zeros((n_row_clusters,n_cols))
        col_clusters_aux = np.zeros((n_col_clusters-1,n_row_clusters))
        col_clusters[col_clusters==col_cluster] = -1
        col_clusters[col_clusters>col_cluster] -= 1
        for row in np.random.choice(np.arange(n_rows),n_rows,replace=False):
            row_cluster = row_clusters[row]
            row_clusters_aux[row_cluster]=np.logical_or(row_clusters_aux[row_cluster],mask[row,:])
        for col in np.random.choice(np.arange(n_cols),n_cols,replace=False):
            if col_clusters[col]==-1:
                fills = (row_clusters_aux[:,col]>col_clusters_aux).sum(axis=1)
                new_col_cluster = np.random.choice(np.where(fills == fills.max())[0])
            else:
                new_col_cluster=col_clusters[col]
            col_clusters_aux[new_col_cluster]=np.logical_or(col_clusters_aux[new_col_cluster],row_clusters_aux[:,col])
            new_col_clusters[col] = new_col_cluster      
        new_col_clusters = new_col_clusters.astype(int)      

        return new_col_clusters


    def _best_selection(self,population1,fitness1,scores1,population2,fitness2,scores2):
        new_population = [([],[]) for i in range(self.pop_size)]
        new_fitness = np.zeros(self.pop_size)*np.nan
        new_scores = np.zeros((self.pop_size,2,self.max_row_clusters,self.max_col_clusters))*np.nan
        selection = np.nan_to_num(fitness1,nan=np.inf) < np.nan_to_num(fitness2,nan=np.inf)
        for i,select in enumerate(selection):
            if select:
                new_population[i] = population1[i]
                new_fitness[i] = fitness1[i]
                new_scores[i,:,:,:] = scores1[i,:,:,:]
            else:
                new_population[i] = population2[i]
                new_fitness[i] = fitness2[i]
                new_scores[i,:,:,:] = scores2[i,:,:,:]
      
        return new_population, new_fitness, new_scores

    def _roullete_selection(self,population1,fitness1,scores1,population2,fitness2,scores2,elitism=True):
        new_population = [([],[]) for i in range(self.pop_size)]
        new_fitness = np.zeros(self.pop_size)*np.nan
        new_scores = np.zeros((self.pop_size,2,self.max_row_clusters,self.max_col_clusters))*np.nan
        probs = np.nan_to_num(np.concatenate((fitness1,fitness2),axis=0))
        np.divide(1,probs, where=probs!=0,out=probs)
        if elitism:
            best = np.nanargmax(probs)
            probs[best] = 0.
        probs = np.argsort(np.argsort(probs)) - np.sum(probs==0)
        probs = np.maximum(probs,0)
        probs = probs/probs.sum()
        selection = np.random.choice(np.arange(probs.size),self.pop_size-1,replace=False,p=probs)
        selection = np.insert(selection,0,best)
        rate = np.mean(selection<self.pop_size)
        for i,select in enumerate(selection):
            if select < self.pop_size:
                new_population[i] = population1[select]
                new_fitness[i] = fitness1[select]
                new_scores[i,:,:,:] = scores1[select,:,:,:]
            else:
                new_population[i-self.pop_size] = population2[select-self.pop_size]
                new_fitness[i-self.pop_size] = fitness2[select-self.pop_size]
                new_scores[i-self.pop_size,:,:,:] = scores2[select-self.pop_size,:,:,:]
        
        
        return rate, new_population, new_fitness, new_scores


    def _print_status(self,gen_count,delta_fitness,population,fitness,rate,infeasible,elapsed_time):
        best = np.nanmin(fitness)
        worst = np.nanmax(fitness)
        mean = np.nanmean(fitness)
        sizes = np.array([(np.max(ind[0])+1)*(np.max(ind[1])+1) for ind in population])
        max_size = sizes.max()
        min_size = sizes.min()
        mean_size = sizes.mean()
        if gen_count==0:
            print('|'.join(x.ljust(11) for x in [
                    'generation','delta fitness (%)','best fitness','worst fitness','mean fitness','max size','min size','mean size','survival rate','infeasible','elapsed time (s)']))

        print('|'.join(x.ljust(11) for x in ['%i' % gen_count,'%.4f' % delta_fitness,'%.4f' % best,'%.4f' % worst,'%.4f'  % mean,'%i'  % max_size,'%i'  % min_size,'%.2f'  % mean_size, '%.4f' % rate ,'%i' % infeasible,'%i' % elapsed_time]))

    def _converge_evoscoal(self,train_data,valid_data,population,max_gen=5,gen_tol=10,tol=0.01,n_jobs=(1,1),verbose=False):
        converged = False
        gen_count = 0
        delta_fitness_arr=np.ones(iter)
        delta_fitness = np.nan
        rate = 1
        infeasible = 0
        elapsed_time = 0
        start = time.time()

        if population is None:
            population, fitness, scores = self._initialize_pop(train_data)
        fitness, scores = self._evaluate_fitness_pop(train_data,valid_data,population,fitness,scores,n_jobs)
        best = np.nanmin(fitness)
        infeasible = np.sum(np.isnan(fitness))
        
        converged = gen_count >= max_gen 

        if verbose:
            self._print_status(gen_count,population,fitness,rate,infeasible,elapsed_time)
        
        while not converged:

            ls_population, ls_fitness, ls_scores = deepcopy(population), deepcopy(fitness), deepcopy(scores)
            ls_population = self._local_search_pop(train_data,valid_data,ls_population,ls_fitness,ls_scores,n_jobs)
            ls_fitness,ls_scores = self._evaluate_fitness_pop(train_data,valid_data,ls_population,ls_fitness,ls_scores,n_jobs)

            population, fitness, scores = self._best_selection(population,fitness,scores,ls_population,ls_fitness,ls_scores) 
            
            mut_population, mut_fitness, mut_scores = deepcopy(population), deepcopy(fitness), deepcopy(scores)
            mut_population = self._muatation_pop(train_data,valid_data,mut_population, mut_fitness, mut_scores,n_jobs)
            mut_fitness, mut_scores = self._evaluate_fitness_pop(train_data,valid_data,mut_population,mut_fitness,mut_scores,n_jobs)

            infeasible = np.sum(np.isnan(fitness)) + np.sum(np.isnan(mut_fitness))
            rate, population, fitness, scores = self._roullete_selection(population,fitness,scores,mut_population,mut_fitness,mut_scores) 
            
            old_best = best
            best = np.nanmin(fitness)
            delta_fitness = (old_best-best)/old_best
            delta_fitness_arr[gen_count%gen_tol] = delta_fitness
            
            gen_count+=1
            converged = (gen_count == max_gen) or (np.nanmax(delta_fitness_arr) < tol)
            elapsed_time = time.time() - start
            if verbose:
                self._print_status(gen_count,delta_fitness,population,fitness,rate,infeasible,elapsed_time)

        if self.fitness_function=='SSE':
            train_matrix, row_features, col_features = train_data
            valid_matrix, _, _ = valid_data
            if self.matrix=='dense':
                train_matrix[np.where(np.invert(np.isnan(valid_matrix)))] = valid_matrix[np.where(np.invert(np.isnan(valid_matrix)))]
            else:
                train_matrix = np.vstack((train_matrix,valid_matrix))
            train_data = (train_matrix,row_features,col_features)

        coclusters = population[np.nanargmin(fitness)]
        models = self._initialize_models(coclusters)

        coclusters, models = self._converge_scoal(train_data,coclusters,models,n_jobs=n_jobs[1],verbose=False)

        return coclusters, models


    def fit(self,target,row_features,col_features):
        np.random.seed(self.random_state) 

        self.n_rows, self.n_cols, self.n_values = row_features.shape[0], col_features.shape[0], target.shape[0]
        self.n_row_features, self.n_col_features  = row_features.shape[1], col_features.shape[1]        
        
        if self.fitness_function=='BIC':
            if self.matrix=='dense':
                matrix = np.zeros((self.n_rows, self.n_cols))*np.nan
                matrix[target[:,0].astype(int),target[:,1].astype(int)] = target[:,2]
            else:
                matrix = target 
            del target
            
            valid_data = None
            train_data = (matrix,row_features,col_features)

        else:
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
            self._cached_fit = self.memory.cache(self._cached_fit, ignore=['self','model','X','y'])

        self.coclusters, self.models = self._converge_evoscoal(train_data,valid_data,None,self.max_gen,self.n_jobs,self.verbose)
        row_clusters, col_clusters = self.coclusters
        self.n_row_clusters, self.n_col_clusters  = np.unique(row_clusters).size, np.unique(col_clusters).size
        self.n_jobs = self.n_jobs[1]
        if self.cache:
            self.memory.clear(warn=False)



  