import numpy as np
from scipy.spatial import distance
from collections import defaultdict

class Codebook: # Maybe inherit from np.ndarray?
    def __init__(self, data=None, size_codebook=None, epsilon=0.01, verbose=False):
        """
        Generate codebook from initialized information.

        :param data: numpy array of shape (n_samples, n_dimensions)
        :param size_codebook: max number of centroids
        :param epsilon: threshold for stopping condition
        :param verbose: print on each iterations
        """
        self.data = data
        self.size_codebook = size_codebook
        self.epsilon = epsilon
        self.verbose = verbose

        if(data is not None and size_codebook is not None):
            self.codebook = self.__generate_codebook()
        else:
            self.codebook = None
    
    def __str__(self) -> str:
        return f'Codebook: {self.codebook}'

    def fit(self, data=None, size_codebook=None, epsilon=None, verbose=None) -> None:
        """
        Generate codebook from initialized information.

        :param data: numpy array of shape (n_samples, n_dimensions)
        :param size_codebook: max number of centroids
        :param epsilon: threshold for stopping condition
        :param verbose: print on each iterations
        """
        if(data is not None):
            self.data = data
        if(size_codebook is not None):
            self.size_codebook = size_codebook
        if(epsilon is not None):
            self.epsilon = epsilon
        if(verbose is not None):
            self.verbose = verbose

        if(self.data is None or self.size_codebook is None):
            raise ValueError('Data and size_codebook must be initialized')
        else:
            self.codebook = self.__generate_codebook()

    def getDistance(self, data) -> float:
        """
        Calculate distance between each data point and the closest centroid.
        This data will be different from the one used to generate the codebook.

        :param data: numpy array of shape (n_samples, n_dimensions)

        :return: distortion: float. Total euclidean distance between each data point and its nearest centroid.
        """
        if(self.codebook is None):
            raise ValueError('Codebook must be initialized')
        else:
            dist = distance.cdist(data, np.array(self.codebook), 'euclidean')
            c_indices = np.argmin(dist, axis=1)
            return self.__distortion(data, c_indices, self.codebook)

    def __generate_codebook(self) -> np.ndarray:
        """
        Cluster data in centers by Lined-Buzo-Gray algorithm.
        :param data: numpy array of shape (n_samples, n_dimensions)
        :param size_codebook: max number of centroids
        :param epsilon: threshold for stopping condition
        :param verbose: print the number of iterations

        :return: codebook: numpy array of shape (size_codebook, n_dimensions)
        """
        n_iterations = 0
        n_samples, n_dimensions = self.data.shape
        codebook = []
        # not sure if abs_weights or rel_weights is needed

        # Initialize codebook with first centroid
        c0 = np.mean(data, axis=0)
        codebook.append(c0)

        # Initialize centroid index for each data point
        c_indices = np.zeros(n_samples)
        
        # Calculate distortion of first centroid
        avg_dist = self.__distortion(self.data, c_indices, codebook)

        # Split centroids until reach the max number of centroids
        while len(codebook) < self.size_codebook:
            codebook = self.__split_codebook(codebook)

            err = 1 + self.epsilon
            while err > self.epsilon:
                # Calculate distance between each data point and each centroid
                dist = distance.cdist(data, np.array(codebook), 'euclidean')

                # Assign each data point to the nearest centroid
                c_indices = np.argmin(dist, axis=1)

                data_near_centroid = defaultdict(list)
                # uniq_centroids = np.unique(c_indices)
                # data_near_centroid = np.zeros((len(uniq_centroids), n_dimensions))
                for c_index in np.unique(c_indices):
                    data_near_centroid[c_index] = data[c_indices == c_index]
                # for i, c_index in enumerate(uniq_centroids):
                #     mask = c_indices == c_index
                #     data_near_centroid[i,:] = data[np.where(c_indices == c_index)]
                
                codebook = self.__update_codebook(data_near_centroid, codebook)

                # Calculate new distance between each data point and each centroid
                new_dist = self.__distortion(self.data, c_indices, codebook)
                err = (avg_dist - new_dist) / avg_dist
                avg_dist = new_dist

                n_iterations += 1
                if self.verbose:
                    print(f'Iteration {n_iterations}: {len(codebook)} centroids, distortion = {avg_dist}')
                    print(f'\tError: {err}')
                    print(f'\tCodebook: {codebook}')
                    print(f'\tIndex for each data point: {c_indices}')
                    print()

        return np.array(codebook)

    def __distortion(self, data, c_index, codebook):
        """
        Calculate distance between each data point and its nearest centroid.
        :param c_index: index of the centroid for each data point
        :param codebook: list of centroids
        :param data: numpy array of shape (n_samples, n_dimensions)

        :return: distortion: float. Total euclidean distance between each data point and its nearest centroid.
        """
        distance = 0

        for i, centroid in enumerate(codebook):
            mask = c_index == i
            distance += np.linalg.norm(data[mask] - centroid, axis=1).sum()

        return distance


    def __split_codebook(self, codebook) -> list:
        """
        Split each centroid.
        :param codebook: list of centroids
        :param epsilon: distance to split centroid

        :return: codebook: new list of centroids. Size will double.
        """
        new_codebook = []

        for centroid in codebook:
            new_codebook.append(centroid + self.epsilon)
            new_codebook.append(centroid - self.epsilon)

        return new_codebook


    def __update_codebook(self, data_near_centroid, codebook) -> list:
        """
        Update each centroid.
        :param data: numpy array of shape (n_samples, n_dimensions)
        :param codebook: list of centroids
        :param uniq_centroids: list of indices of centroids

        :return: codebook: codebook with adjusted centroid positions
        """
        for i, centroid in enumerate(codebook):
            if len(data_near_centroid[i]) > 0:  # if there are no data points near the centroid, don't update it
                codebook[i] = np.mean(data_near_centroid[i], axis=0)
        
        return codebook



# Testing
if __name__ == '__main__':
    #data = np.random.rand(100, 2)
    data = np.array([[i,j] for i in range(1,6) for j in range(1,6,2)])
    print(data.shape)
    print(data)
    print()
    cb = Codebook(data, 16, verbose=True)
    print(cb)
    print(cb.getDistance(data))