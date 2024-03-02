import numpy as np
import pandas as pd
from scipy.spatial import distance

def generate_codebook(data, size_codebook, epsilon=0.01, verbose=False):
    """
    Cluster data in centers by Lined-Buzo-Gray algorithm.
    :param data: numpy array of shape (n_samples, n_dimensions)
    :param size_codebook: max number of centroids
    :param epsilon: threshold for stopping condition
    :param verbose: print the number of iterations

    :return: codebook: numpy array of shape (size_codebook, n_dimensions)
    """

    n_samples, n_dimensions = data.shape
    codebook = []
    # not sure if abs_weights or rel_weights is needed

    # Initialize codebook with first centroid
    c0 = np.mean(data, axis=0)
    codebook.append(c0)

    # Initialize centroid index for each data point
    c_indices = np.zeros(n_samples)
    
    # Calculate distortion of first centroid
    dist_c0 = distortion(c_indices, codebook, data)

    # Split centroids until reach the max number of centroids
    while len(codebook) < size_codebook:
        codebook = split_codebook(codebook, epsilon)

        err = 1 + epsilon
        while err > epsilon:
            # Calculate distance between each data point and each centroid
            dist = distance.cdist(data, np.array(codebook), 'euclidean')

            # Assign each data point to the nearest centroid
            c_indices = np.argmin(dist, axis=1)

            uniq_centroids = np.unique(c_indices)
            data_near_centroid = np.zeros((len(uniq_centroids), n_dimensions))
            for i, c_index in enumerate(uniq_centroids):
                mask = c_indices == c_index
                data_near_centroid[i,:] = data[np.where(c_indices == c_index)]
            
            update_codebook(data_near_centroid, codebook, uniq_centroids)



    
    if verbose:
        print(f'Initial centroid: {c0}')
        print(f'Initial distortion: {dist_c0}')

def distortion(c_index, codebook, data):
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

def split_codebook(codebook, epsilon):
    """
    Split each centroid.
    :param codebook: list of centroids
    :param epsilon: distance to split centroid

    :return: codebook: new list of centroids. Size will double.
    """
    new_codebook = []
    for centroid in codebook:
        new_codebook.append(centroid + epsilon)
        new_codebook.append(centroid - epsilon)
    return new_codebook

def update_codebook(data, codebook, uniq_centroids):
    """
    Update each centroid.
    :param data: numpy array of shape (n_samples, n_dimensions)
    :param codebook: list of centroids
    :param uniq_centroids: list of indices of centroids

    :return: codebook: codebook with adjusted centroid positions
    """
    # for i, centroid in enumerate(uniq_centroids):
    #     codebook[centroid] = np.mean(data[i,:], axis=0

    pass

if __name__ == '__main__':
    #data = np.random.rand(100, 2)
    data = np.array([[i,j] for i in range(1,6) for j in range(1,6,2)])
    #print(data)
    print(data.shape)
    codebook = generate_codebook(data, 10, verbose=True)