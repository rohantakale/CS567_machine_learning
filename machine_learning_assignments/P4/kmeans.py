import numpy as np
# This part of assignment is discussed with Shubhi Bharal (USC ID: 7079131249).

class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
            Note: Number of iterations is the number of time you update the assignment
        '''

        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        x = np.array(x)
        random_idx = np.random.choice(N, size=self.n_cluster)
        muk = x[random_idx]
        j_old = 0
        distance = np.empty((N, self.n_cluster))
        gamma = np.zeros((N, self.n_cluster))	# gamma

        count = 0
        while count < self.max_iter:
        	j_new = 0
        	for i in range(len(muk)):
        		distance[:, i] = np.linalg.norm(np.subtract(muk[i], x), axis=1) ** 2
        	gamma = np.eye(len(muk))[np.argmin(distance, axis=1)]

        	for k in range(len(muk)):
        		j_new += np.dot(gamma[:, k], np.linalg.norm(np.subtract(muk[k], x), axis=1) ** 2)
        	j_new /= N

        	if np.absolute(j_new - j_old) < self.e:
        		break
        	j_old = j_new

        	for n in range(len(muk)):
        		muk[n] = np.dot(gamma[:, n], x) / np.sum(gamma[:, n])
        	count += 1

        return (muk, np.argmax(gamma, axis=1), count)
        # DONOT CHANGE CODE BELOW THIS LINE

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float) 
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by 
                    majority voting ((N,) numpy array) 
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        clf_kmeans = KMeans(self.n_cluster, self.max_iter, self.e)
        centroids, membership, count = clf_kmeans.fit(x)
        centroid_labels = []

        for i in range(len(centroids)):
            labels = []
            for memb_i in range(len(membership)):
                if membership[memb_i] == i:
                    labels.append(y[memb_i])
            bincnt = np.bincount(np.array(labels))
            centroid_labels.append(np.argmax(bincnt))

        centroid_labels = np.array(centroid_labels)

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        preds = []

        for i in range(len(x)):
            distnc = []

            for k in range(len(self.centroids)):
                distnc.append(np.linalg.norm(x[i] - self.centroids[k]))
            dist = np.array(distnc)
            preds.append(self.centroid_labels[np.argmin(dist)])

        labels = np.array(preds)
        # DONOT CHANGE CODE BELOW THIS LINE
        return labels

