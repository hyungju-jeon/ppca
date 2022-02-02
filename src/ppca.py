import numpy as np
from numpy.linalg import inv
from numpy import trace as tr

class PPCA(object):
	"""	EM-based PPCA """

	def __init__(self, q=2, sigma=1.0, max_iter=50):
		# OBSERVED VARIABLE
		# y : observed variable (data)
		self.y = None
		# d : data dimentionality
		self.d = 0
		# N : number of data 
		self.N = 0
		# mu : data mean (y_bar)
		self.mu = None
		# Latent VARIABLE
		# q : latent variable dimensionality 
		self.q = q
		# sigma : standard deviation of the noise
		self.sigma = sigma
		# W : d x q matrix relating observed vs. latent variable 
		self.W = None
		# EM properties
		# max_iter : maximum iterations
		self.max_iter = max_iter
			
	def fit(self, data, max_iter=None, q=None, verbose=False):
		""" 
		Perform PPCA to find latent space and transform

		Parameters
		----------
		data : ndarray
			Observed variables that are being tranformed
		max_iter : int (optional)
			maximum EM iteration 
		q : int (optional)
			latent variable dimensionality 
		verbose : logical (optional)
			Display iteration 
		"""
		self.y = data
		self.d = data.shape[0]
		self.N = data.shape[1]
		self.mu = np.mean(data, axis=0)
		if max_iter is not None:
			self.max_iter = max_iter
		if q is not None:
			self.q = q
		print('Starting EM algorithm')
		[W, y, d, q, N, mu, sigma] = [self.W, self.y, self.d, self.q, self.N, self.mu, self.sigma]
		if W is None:
			W = np.random.rand(self.d, self.q)

		for i in range(self.max_iter):
			if verbose:
				if self.max_iter > 10:
					if i%10==0: print(f'Running iter {i}') 
				else:
					print(f'Running iter {i}')
			# Running E-step
			M = (W.T).dot(W) + sigma * np.eye(q) # M : q x q
			invM = inv(M) # invM : q x q
			EZ = invM.dot(W.T).dot((y - mu)) # EZ : q x N
			EZZ = sigma*invM + EZ.dot(EZ.T) # EZZ : q x q
			# Running M-step	
			W = ((y - mu).dot(EZ.T)).dot(inv(EZZ))
			sigma = np.linalg.norm(y - mu) - 2*tr((EZ.T).dot(W.T).dot((y - mu))) + tr(EZZ.dot((W.T).dot(W)))
			sigma = (1/(N*d))*sigma
		self.W = W
		self.sigma = sigma


	def transform(self, data=None):
		""" 
		Transform observed variable onto latent space 

		Parameters
		----------
		data : ndarray (optional)
			Observed variables that are being tranformed

		Returns
		-------
		x : ndarray
			transformed latent variable
		"""
		if data is None:
			y = self.y
		[W, q, mu, sigma] = [self.W, self.q, self.mu, self.sigma]
		M = (W.T).dot(W) + sigma * np.eye(q)
		invM = inv(M)
		x = invM.dot(W.T).dot(y - mu)
		return x

	def reconstruct(self, x=None):
		""" 
		Reconstruct signal from latent variable 

		Parameters
		----------
		data : ndarray (optional)
			latent variable

		Returns
		-------
		y : ndarray
			Reconstructed original signal
		"""
		if x is None:
			x = self.transform()
		[W, d, N, mu, sigma] = [self.W, self.d, self.N, self.mu, self.sigma]
		y = W.dot(x) + mu
		for i in range(N):
			eps = np.random.normal(0, sigma, d)
			y[:, i] += eps
		return y