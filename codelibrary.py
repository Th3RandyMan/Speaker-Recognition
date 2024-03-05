import numpy as np
from glob import glob
from pathlib import Path
from scipy.io import wavfile
from codebook import Codebook
from collections import defaultdict
from processing import feature_extraction

class CodeLibrary(dict):
    """
    Class to represent a library of codebooks.
    """
    def __init__(self, codebooks=None):
        """
        Initialize the CodeLibrary with a dictionary of codebooks.

        :param codebooks: dictionary of Codebook objects
        """
        for codebook in codebooks if codebooks is not None else []:
            self.addCodebook(codebook)

    def __str__(self) -> str:
        return f'CodeLibrary: {self}'

    def addCodebook(self, codebook: Codebook) -> None:
        """
        Add a codebook to the library.

        :param codebook: Codebook object
        """
        self[codebook.name] = codebook

    def removeCodebook(self, codebook: Codebook) -> None:
        """
        Remove a codebook from the library.

        :param codebook: Codebook object
        """
        del self[codebook.name]

    def getCodebook(self, name: str) -> Codebook:
        """
        Get a codebook from the library.

        :param name: name of the codebook
        :return: codebook: Codebook object
        """
        return self[name]

    def getDistance(self, data: np.ndarray) -> list[float]:
        """
        Get the distance of the data to each codebook in the library.

        :param data: numpy array of shape (n_samples, n_dimensions)
        :return: distances: list of floats
        """
        distances = []
        for name, codebook in self.items():
            distances.append(codebook.getDistance(data))
        return distances

    def getClosestCodebookName(self, data: np.ndarray) -> str:
        """
        Get the closest codebook to the data in the library.

        :param data: numpy array of shape (n_samples, n_dimensions)
        :return: str: Codebook name
        """
        distances = self.getDistance(data)
        return list(self.keys())[np.argmin(distances)]

    def getClosestCodebook(self, data: np.ndarray) -> Codebook:
        """
        Get the closest codebook to the data in the library.

        :param data: numpy array of shape (n_samples, n_dimensions)
        :return: codebook: Codebook object
        """
        return self[self.getClosestCodebookName(data)]
    
    def getClosestCodebookDistance(self, data: np.ndarray) -> float:
        """
        Get the distance of the data to the closest codebook in the library.

        :param data: numpy array of shape (n_samples, n_dimensions)
        :return: float: distance
        """
        return self.getClosestCodebook(data).getDistance(data)
    
    def predict(self, filename: str, N: int = 1024, M: int = 512, n_mfcc: int = 20) -> str:
        """
        Predict the class of the audio file.

        :param filename: str. Path to the audio file
        :return: str. Name of the predicted class
        """
        sampling_rate, audio_data = wavfile.read(filename)
        return self.getClosestCodebookName(feature_extraction(audio_data, N, M, sampling_rate, n_mfcc))
    
    def fillLibrary(self, data_folder: str) -> None:
        """
        Fill the library with codebooks from a folder.

        :param data_folder: str. Path to the folder containing the codebooks
        """
        for filename in glob(f'{data_folder}\*.npy'):
            codebook = Codebook()
            codebook.load(filename)
            self.addCodebook(codebook)
    
    def createLibrary(self, data_folder: str = None, N: int = 1024, M: int = 512, n_mfcc: int = 20, size_codebook: int = 32, epsilon: float = 0.01, verbose: bool = False) -> None:
        """
        Create the library from a folder of audio files.

        :param data_folder: str. Path to the folder containing the audio files
        :param N: int. The number of samples in each frame
        :param M: int. The number of samples to move between frames
        :param n_mfcc: int. The number of MFCC coefficients to return
        :param size_codebook: int. Number of centroids
        :param epsilon: float. Threshold for stopping condition
        :param verbose: bool. Print on each iterations
        """
        if(data_folder is None):
            data_folder = Path().resolve() / "Codebooks"

        for filename in glob(f'{data_folder}\*.wav'):
            name = filename.split('\\')[-1][:-4]
            sampling_rate, audio_data = wavfile.read(filename)

            # Extract features
            mfcc_features = feature_extraction(np.array(audio_data), N, M, sampling_rate, n_mfcc)
            
            # Create codebook
            codebook = Codebook(mfcc_features, size_codebook=size_codebook, epsilon=epsilon, verbose=verbose)
            codebook.save(name)
            self.addCodebook(codebook)
    
if __name__ == '__main__':
    N = 1024    # Number of samples in each frame
    M = 512     # Number of samples to move between frames
    n_mfcc = 20 # Number of MFCC coefficients to return

    size_codebook = 32  # Number of centroids
    epsilon = 0.01      # Threshold for stopping condition
    verbose = False     # Print on each iterations

    # Create codebooks for the library
    data_folder = Path().resolve() / "Audio Files"

    twelve_train_files = glob(f'{data_folder}\Twelve Train\*.wav')
    zero_train_files = glob(f'{data_folder}\Zero Train\*.wav')

    # Create codebooks
    twelve_codelibrary = CodeLibrary()
    zero_codelibrary = CodeLibrary()

    for filename in twelve_train_files:
        name = filename.split('\\')[-1][:-4]
        sampling_rate, audio_data = wavfile.read(filename)

        # Extract features
        mfcc_features = feature_extraction(np.array(audio_data), N, M, sampling_rate, n_mfcc)
        
        # Create codebook
        codebook = Codebook(mfcc_features, size_codebook=size_codebook, epsilon=epsilon, verbose=verbose)
        codebook.save(name)
        twelve_codelibrary.addCodebook(codebook)
    
    for filename in zero_train_files:
        name = filename.split('\\')[-1][:-4]
        sampling_rate, audio_data = wavfile.read(filename)

        # Extract features
        mfcc_features = feature_extraction(np.array(audio_data), N, M, sampling_rate, n_mfcc)
        
        # Create codebook
        codebook = Codebook(mfcc_features, size_codebook=size_codebook, epsilon=epsilon, verbose=verbose)
        codebook.save(name)
        zero_codelibrary.addCodebook(codebook)

    # Test the library
    print(zero_codelibrary.getClosestCodebookName(feature_extraction(audio_data, N, M, sampling_rate, n_mfcc)))
