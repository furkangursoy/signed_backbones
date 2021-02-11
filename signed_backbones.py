""" Python package for extracting signed backbones of intrinsically dense weighted networks."""
__version__ = '0.9.5'

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

def extract(edgelist, directed = True, significance_threshold = 3.291, vigor_threshold = 0, return_weights = False, max_iteration = 100, precision = 10e-8):
	"""Extract the signed backbones of weighted networks.
	
    Signed backbones are extracted based on the significance filter and vigor filter as described in the following paper. Please cite it if you find this software useful in your work.
            
    Furkan Gursoy and Bertan Badur. "Extracting the signed backbone of intrinsically dense weighted networks." (2020). https://https://arxiv.org/abs/2012.05216
    
    Parameters
    ----------
    edgelist : pandas DataFrame
        A pandas DataFrame of shape (n_original_edges, 3).
        First two columns contain node pairs, and the third column contains the edge weights. 
        If directed = True, columns should be in this order: source node, target node, edge weight.
        
    directed : bool (optional, default=True)
        Whether the input network is directed.
        
    significance_threshold : float or tuple (optional, default=3.291)
        Threshold for the significance filter.
        If float, a single nonnegative value, e.g., 1.23.
        If tuple, a tuple of nonpositive and nonnegative values, e.g., (-1.23, 4.56).
        1.23 is equivalent to (-1.23, 1.23).
        
    vigor_threshold : float or tuple (optional, default=0)
        Threshold for the vigor filter.
        If float, a single nonnegative value in the range [0, 1], e.g., 0.33.
        If tuple, a tuple of nonpositive and nonnegative values in the ranges [-1, 0] and [0, 1], e.g., (-0.5, 0.3).
        0.33 is equivalent to (-0.33, 0.33).
        
    return_weights : bool (optional, default=True)
        Whether the returned backbone should contain the signed link weights that show the intensity of signed links.    
        
    max_iteration : int (optional, default=100)
    	Maximum number of iterations to be used in the Iterational Proportional Fitting Procedure.
    	
    precision : float or tuple (optional, default=10e-8) 
    	A small epsilon value to be used in comparison with zero values due to numerical precision issues. Can be left as default.


    Returns
    -------
    
    A numpy ndarray of shape (n_backbone_edges, 3) containing the edges for the extracted backbone.
    	First two columns contain node pairs, and the third column contains the edge sign. 
        If directed = True, columns are in this order: source node, target node, edge sign.
        If return_weights = True, signed edge weights are returned instead of edge sign.
        
        
    Notes
    -------
    For usage examples, please visit the project's main page at https://github.com/furkangursoy/signed_backbones
	"""
	
	if directed:
		edgelist = pd.DataFrame(edgelist.values)
	else:
		edgelist = pd.DataFrame(np.concatenate([edgelist.values, edgelist.iloc[:, [1,0,2]].values]))

	edgedf = calculateStatistics(edgelist, max_iteration, precision)
	 
	
	if isinstance(significance_threshold, tuple):
		if significance_threshold[0] > 0:
			raise Exception('If significance_threshold is a tuple, the first element should be non-positive.')
		if significance_threshold[1] < 0:
			raise Exception('If significance_threshold is a tuple, the second element should be non-negative.')	
		
		significance_filter_index = np.logical_or(edgedf.StdDist <= significance_threshold[0] , edgedf.StdDist >= significance_threshold[1])
		
	elif np.isscalar(significance_threshold):
		significance_filter_index = np.abs(edgedf.StdDist) >= significance_threshold
	else:
		raise Exception('significance_threshold should be a scalar or a tuple.')
        
        
	if isinstance(vigor_threshold, tuple):
		if vigor_threshold[0] > 0:
			raise Exception('If vigor_threshold is a tuple, the first element should be non-positive.')
		if vigor_threshold[1] < 0:
			raise Exception('If vigor_threshold is a tuple, the second element should be non-negative.')	
		
		vigor_filter_index = np.logical_or(edgedf.LiftScore <= vigor_threshold[0] , edgedf.LiftScore >= vigor_threshold[1])
		
	elif np.isscalar(vigor_threshold):
		vigor_filter_index = np.abs(edgedf.LiftScore) >= vigor_threshold
	else:
		raise Exception('vigor_threshold should be a scalar or a tuple.')

	filter_index = np.logical_and(vigor_filter_index, significance_filter_index)
	
	edges = edgedf.loc[filter_index, :]
	if not directed:
		edges.loc[:,'pair'] = edges.iloc[:,[0,1]].apply(frozenset, axis=1)
		edges.loc[:,'absLiftScore'] = np.abs(edges.loc[:,'LiftScore'])
		edges = edges.sort_values(by=['pair', 'absLiftScore'], ascending=False)
		edges = edges.drop_duplicates(subset='pair', keep='first').sort_index().reset_index(drop=True)
	
	if return_weights:
		edges = edges.loc[:, ['i', 'j', 'LiftScore']].to_numpy()
	else:
		edges = edges.loc[:, ['i', 'j', 'Sign']].to_numpy()

    
	print('{} edges are retained.'.format(len(edges)))        
    
	return edges
	
	
	

def __calculateStatistics(edgelist, max_iteration, precision):
	edgedf = pd.DataFrame(edgelist.values, columns =  ['i', 'j', 'Wij'])

	n_self_loops = (edgedf['i'] == edgedf['j']).sum() 
	
	if n_self_loops > 0:
		edgedf = edgedf.loc[edgedf['i'] != edgedf['j'],:]
		print("{} self loops are removed.".format(n_self_loops))
		
	# check for duplicate rows   
	if edgedf.duplicated(subset=['i', 'j']).any():
		edgedf = edgedf.groupby(by = ['i', 'j']).sum().reset_index()
		print("Duplicate edges are identified. The edges are merged by summing up their weights.")
	
    #add links with 0 weights to the edge list
	nodes = np.unique(edgedf[['i', 'j']]) # list of nodes
	adj = pd.DataFrame(.0, index=nodes, columns=nodes) # create an empty adjacency matrix
	f = adj.index.get_indexer # to fill in adjacency matrix values in the next line
	adj.values[f(edgedf.i), f(edgedf.j)] = edgedf.Wij.values
	edgedf = pd.DataFrame(adj.transpose().unstack().reset_index().values, columns = ['i', 'j', 'Wij'])

    
    
	edgedf = pd.merge(edgedf, 
					 edgedf[['i','Wij']].groupby('i', as_index=False).sum().rename(columns ={'Wij':'Wi.'}),
					 left_on = 'i', right_on = 'i')
	
	edgedf = pd.merge(edgedf, 
					 edgedf[['j','Wij']].groupby('j', as_index=False).sum().rename(columns ={'Wij':'W.j'}),
					 left_on = 'j', right_on = 'j')
    
	edgedf = pd.merge(edgedf, 
			edgedf[['j', 'W.j']].drop_duplicates(), 
			how = 'left', left_on = 'i', right_on = 'j', 
			suffixes=('', '_y') ).rename(columns ={'W.j_y':'W.i'}).drop(columns = ['j_y'])

	edgedf['W..'] = edgedf['Wij'].sum()
	
    #prepare the matrix for passing to getNullMatrix
	nodes = np.unique(edgedf[['i', 'j']]) # list of nodes
	adj = pd.DataFrame(.0, index=nodes, columns=nodes) # create an empty adjacency matrix
	f = adj.index.get_indexer # to fill in adjacency matrix values in the next line
	adj.values[f(edgedf.i), f(edgedf.j)] = edgedf.Wij.values

	edgedf = pd.merge(edgedf, 
			pd.DataFrame(getNullMatrix(adj.to_numpy(), max_iteration, precision), index=adj.index, columns=adj.columns).unstack().reset_index(),
			 left_on = ['i', 'j'], right_on = ['level_1', 'level_0'], how = 'left').rename(columns={0:'Nij'})
    

	N = edgedf['W..'] - edgedf['W.i'] 
	K = np.where(edgedf['W.j'] == 0, precision, edgedf['W.j'])
	n = np.where(edgedf['Wi.'] == 0, precision, edgedf['Wi.'])
    
    

	edgedf['Var'] = n * (K/N) * ((N-K)/N) * ((N-n)/(N-1))
	edgedf['Std'] = edgedf['Var'] ** .5
	edgedf['StdDist'] = (edgedf['Wij'] - edgedf['Nij'])/edgedf['Std']
	edgedf['Lift'] = edgedf['Wij'] / np.where(edgedf['Nij']== 0, 1, edgedf['Nij'])
	edgedf['LiftScore'] = (edgedf['Lift'] - 1) / (edgedf['Lift'] + 1)
	edgedf['Sign'] = np.sign(edgedf['LiftScore'])
	edgedf = edgedf.loc[edgedf.i != edgedf.j, :]
	return edgedf


def __getNullMatrix(matrix, max_iteration, precision):
		
	n = len(matrix) # number of nodes

	marginal_row = np.sum(matrix, axis = 1) # row totals
	marginal_column = np.sum(matrix, axis = 0) # column totals

	
	marginal_row = np.where(marginal_row==0, precision**2, marginal_row)
	marginal_column = np.where(marginal_column==0, precision**2, marginal_column)


	prior = (np.array([marginal_column]*n) * np.array([marginal_row] *n).T ) / np.sum(matrix) # initial Pij = (Ai. * A.j) / A.. where "." indicates sum over all respective indices
	np.fill_diagonal(prior, 0) # set diagonals to zero since self-loops are not allowed
	
	iteration = 0 # iteration counter
	null = prior.copy()
	while iteration < max_iteration:
		
		# row scaling
		row_scaler = marginal_row / np.sum(null, axis=1)
		null = null * np.array([row_scaler]*n).T
		
		# column scaling
		column_scaler = marginal_column / np.sum(null, axis=0)
		null = null * np.array([column_scaler]*n)

		iteration +=1
		
		# 
		MAE_row = np.mean(np.abs(marginal_row - np.sum(null, axis = 1)))
		MAE_column = np.mean(np.abs(marginal_column - np.sum(null, axis = 0)))
		
		# end the optimization if the desired precision is achieved earlier than the max_iteration
		if ( MAE_row < precision and MAE_column < precision):
			print('Iterative fitting procedure converged at iteration {}.'.format(iteration))
			return null
			
		
	print('Iterative Fitting Procedure ended at iteration {} with Mean Absolute Error (MAE) {} for row totals and MAE {} for column totals.'.format(iteration, MAE_row, MAE_column))
		
	return null
	
	
	
def calculateStatistics(edgelist, max_iteration, precision):
	edgedf = pd.DataFrame(edgelist.values, columns =  ['i', 'j', 'Wij'])

	n_self_loops = (edgedf['i'] == edgedf['j']).sum() 
	
	if n_self_loops > 0:
		edgedf = edgedf.loc[edgedf['i'] != edgedf['j'],:]
		print("{} self loops are removed.".format(n_self_loops))
		
	# check for duplicate rows   
	if edgedf.duplicated(subset=['i', 'j']).any():
		edgedf = edgedf.groupby(by = ['i', 'j']).sum().reset_index()
		print("Duplicate edges are identified. The edges are merged by summing up their weights.")
	
    #add links with 0 weights to the edge list
	nodes = np.unique(edgedf[['i', 'j']]) # list of nodes
	adj = pd.DataFrame(.0, index=nodes, columns=nodes) # create an empty adjacency matrix
	f = adj.index.get_indexer # to fill in adjacency matrix values in the next line
	adj.values[f(edgedf.i), f(edgedf.j)] = edgedf.Wij.values
	edgedf = pd.DataFrame(adj.transpose().unstack().reset_index().values, columns = ['i', 'j', 'Wij'])

    
    
	edgedf = pd.merge(edgedf, 
					 edgedf[['i','Wij']].groupby('i', as_index=False).sum().rename(columns ={'Wij':'Wi.'}),
					 left_on = 'i', right_on = 'i')
	
	edgedf = pd.merge(edgedf, 
					 edgedf[['j','Wij']].groupby('j', as_index=False).sum().rename(columns ={'Wij':'W.j'}),
					 left_on = 'j', right_on = 'j')
    
	edgedf = pd.merge(edgedf, 
			edgedf[['j', 'W.j']].drop_duplicates(), 
			how = 'left', left_on = 'i', right_on = 'j', 
			suffixes=('', '_y') ).rename(columns ={'W.j_y':'W.i'}).drop(columns = ['j_y'])

	edgedf['W..'] = edgedf['Wij'].sum()
	
    #prepare the matrix for passing to getNullMatrix
	nodes = np.unique(edgedf[['i', 'j']]) # list of nodes
	adj = pd.DataFrame(.0, index=nodes, columns=nodes) # create an empty adjacency matrix
	f = adj.index.get_indexer # to fill in adjacency matrix values in the next line
	adj.values[f(edgedf.i), f(edgedf.j)] = edgedf.Wij.values

	edgedf = pd.merge(edgedf, 
			pd.DataFrame(getNullMatrix(adj.to_numpy(), max_iteration, precision), index=adj.index, columns=adj.columns).unstack().reset_index(),
			 left_on = ['i', 'j'], right_on = ['level_1', 'level_0'], how = 'left').rename(columns={0:'Nij'})
    

	N = edgedf['W..'] - edgedf['W.i'] 
	K = np.where(edgedf['W.j'] == 0, precision, edgedf['W.j'])
	n = np.where(edgedf['Wi.'] == 0, precision, edgedf['Wi.'])
    
    

	edgedf['Var'] = n * (K/N) * ((N-K)/N) * ((N-n)/(N-1))
	edgedf['Std'] = edgedf['Var'] ** .5
	edgedf['StdDist'] = (edgedf['Wij'] - edgedf['Nij'])/edgedf['Std']
	edgedf['Lift'] = edgedf['Wij'] / np.where(edgedf['Nij']== 0, 1, edgedf['Nij'])
	edgedf['LiftScore'] = (edgedf['Lift'] - 1) / (edgedf['Lift'] + 1)
	edgedf['Sign'] = np.sign(edgedf['LiftScore'])
	edgedf = edgedf.loc[edgedf.i != edgedf.j, :]
	return edgedf


def getNullMatrix(matrix, max_iteration, precision):
		
	n = len(matrix) # number of nodes

	marginal_row = np.sum(matrix, axis = 1) # row totals
	marginal_column = np.sum(matrix, axis = 0) # column totals

	
	marginal_row = np.where(marginal_row==0, precision**2, marginal_row)
	marginal_column = np.where(marginal_column==0, precision**2, marginal_column)


	prior = (np.array([marginal_column]*n) * np.array([marginal_row] *n).T ) / np.sum(matrix) # initial Pij = (Ai. * A.j) / A.. where "." indicates sum over all respective indices
	np.fill_diagonal(prior, 0) # set diagonals to zero since self-loops are not allowed
	
	iteration = 0 # iteration counter
	null = prior.copy()
	while iteration < max_iteration:
		
		# row scaling
		row_scaler = marginal_row / np.sum(null, axis=1)
		null = null * np.array([row_scaler]*n).T
		
		# column scaling
		column_scaler = marginal_column / np.sum(null, axis=0)
		null = null * np.array([column_scaler]*n)

		iteration +=1
		
		# 
		MAE_row = np.mean(np.abs(marginal_row - np.sum(null, axis = 1)))
		MAE_column = np.mean(np.abs(marginal_column - np.sum(null, axis = 0)))
		
		# end the optimization if the desired precision is achieved earlier than the max_iteration
		if ( MAE_row < precision and MAE_column < precision):
			print('Iterative fitting procedure converged at iteration {}.'.format(iteration))
			return null
			
		
	print('Iterative Fitting Procedure ended at iteration {} with Mean Absolute Error (MAE) {} for row totals and MAE {} for column totals.'.format(iteration, MAE_row, MAE_column))
		
	return null