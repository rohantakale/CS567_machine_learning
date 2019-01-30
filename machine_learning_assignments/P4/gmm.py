import numpy as np
from kmeans import KMeans
# This part of assignment is discussed with 
# Kabir Chopra (id: 7166329083) and Shubhi Bharal (id: 7079131249).

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE

            # - initialize means using k-means clustering
            clf = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
            self.means, membership, cnt_runs = clf.fit(x)
            r = membership

            #compute variance and pi_k
            self.variances = np.zeros((self.n_cluster,D,D))
            self.pi_k = np.zeros(self.n_cluster)
            for k in range(self.n_cluster):
                indices = np.where(r == k)
                var_numerator = np.zeros((D,D))
                for i in indices[0]:
                    subtn = (x[i] - self.means[k]).reshape((D,1))
                    var_numerator += np.dot(subtn, subtn.T)
                sumk = indices[0].shape[0]
                self.variances[k] = var_numerator / sumk
                self.pi_k[k] = sumk / N

            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE

            # - initialize means randomly
            self.means = np.random.rand(self.n_cluster,D)

            # - initialize variance to be identity and pi_k to be uniform
            self.variances = np.zeros((self.n_cluster,D,D))
            self.pi_k = np.zeros(self.n_cluster)
            for k in range(self.n_cluster):
                self.variances[k] = np.identity(D)
                self.pi_k[k] = 1 / self.n_cluster
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        # DONOT MODIFY CODE ABOVE THIS LINE
        
        def rank_check(varnc):
            var = np.copy(varnc)
            while (np.linalg.matrix_rank(var) < D):
                var += 0.001 * np.identity(D)
            return var

        def calc_gamma(mu, eta, pi):
            determt = np.linalg.det(eta)
            denomtr = np.sqrt((2 * np.pi) ** D * determt)
            term1 = np.exp(-0.5 * np.sum(np.multiply(np.dot(x - mu, np.linalg.inv(eta)), x - mu), axis=1)) / denomtr
            return pi * term1

        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Expectation - Maximization (EM) Algorithm 
        iter = 0
        log_like = -np.inf
        gamma = np.zeros((N, self.n_cluster))
        while iter < self.max_iter:
            # E step
            for k in range(self.n_cluster):
                means_k = self.means[k]
                var_k = rank_check(self.variances[k])
                gamma[:, k] = calc_gamma(means_k, var_k, self.pi_k[k])
            log_like_new = np.sum(np.log(np.sum(gamma, axis=1)))
            gamma = (gamma.T / np.sum(gamma, axis=1)).T
            Nk = np.sum(gamma, axis=0)

            # M step
            for k in range(self.n_cluster):
                self.means[k] = np.transpose(np.sum(gamma[:, k] * np.transpose(x), axis=1)) / Nk[k]
                self.variances[k] = np.dot(np.multiply(np.transpose(x - self.means[k]), gamma[:, k]), x - self.means[k]) / Nk[k]
            self.pi_k = Nk / N
            if (np.abs(log_like - log_like_new) <= self.e):
                break
            log_like = log_like_new
            iter += 1

        # - Return the number of E/M-Steps (iterations) executed 
        return iter
        # DONOT MODIFY CODE BELOW THIS LINE

		
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        samples = []
        self.pi_k /= np.sum(self.pi_k)
        random_k = np.random.choice(self.n_cluster, size=N, p=self.pi_k)
        for k in random_k:
            samples.append(np.random.multivariate_normal(self.means[k], self.variances[k]))
        samples= np.array(samples)
        # DONOT MODIFY CODE BELOW THIS LINE
        return samples        

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k    
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        
        N, D = x.shape

        for k in range(len(self.means)):
            while True:
                if np.linalg.matrix_rank(self.variances[k]) != self.variances[k].shape[0]:
                    self.variances[k] += 0.001 * np.identity(self.variances[k].shape[0])
                else:
                    break
        def calc_gammas(mu, eta, pi):
            determt = np.linalg.det(eta)
            denmtr = np.sqrt((2 * np.pi) ** D * determt)
            term = np.exp(-0.5 * np.sum(np.multiply(np.dot(x - mu, np.linalg.inv(eta)), x - mu), axis=1)) / denmtr
            return pi * term

        gamma = np.zeros((N, self.n_cluster))
        for k in range(self.n_cluster):
            means_k = means[k]
            gamma[:, k] = calc_gammas(means_k, self.variances[k], self.pi_k[k])

        log_likelihood = float(np.sum(np.log(np.sum(gamma, axis=1))))
        # DONOT MODIFY CODE BELOW THIS LINE
        return log_likelihood

    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            D = variance.shape[1]
            while np.linalg.matrix_rank(variance) != D:
                variance = variance + 0.001 * np.identity(D)
            self.inv = np.linalg.inv(variance)
            self.c = np.power(2 * np.pi, D) * np.linalg.det(variance)
            
            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            sub = x - self.mean
            p = np.exp(np.matmul(np.matmul(-0.5 * sub, self.inv), sub.T / np.sqrt(self.c)))
            # DONOT MODIFY CODE BELOW THIS LINE
            return p
