import numpy as np
from glob import glob
from pathlib import Path
from scipy.io import wavfile
from codebook import Codebook
from collections import defaultdict
from processing import feature_extraction
import re
import os

class CodeLibrary(dict):
    """
    Class to represent a library of codebooks.
    """
    def __init__(self, codebooks: Codebook=None):
        """
        Initialize the CodeLibrary with a dictionary of codebooks.

        :param codebooks: dictionary of Codebook objects
        """
        for codebook in codebooks if codebooks is not None else []:
            self.addCodebook(codebook)

    def __str__(self) -> str:
        return f'CodeLibrary: {self.values}'

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
    
    def predict(self, filename: str, N: int = 1024, M: int = 512, n_mfcc: int = 20, window: str = 'hamming', beta: float = 14) -> str:
        """
        Predict the class of the audio file.

        :param filename: str. Path to the audio file
        :return: str. Name of the predicted class
        """
        sampling_rate, audio_data = wavfile.read(filename)
        return self.getClosestCodebookName(feature_extraction(audio_data, N, M, sampling_rate, n_mfcc, window, beta))
    
    def getAccuracy(self, test_files: list[str], N: int = 1024, M: int = 512, n_mfcc: int = 20, window: str = 'hamming', beta: float = 14) -> float:
        """
        Get the accuracy of the library on a set of test files.

        :param test_files: list of str. Paths to the test audio files
        :return: float. Accuracy
        """
        correct = 0
        for filename in test_files:
            number_string = re.search(r'\d+', os.path.basename(filename))
            test_number = int(number_string.group()) if number_string else None


            predicted_string = self.predict(filename, N, M, n_mfcc, window, beta)

            predicted_number_string = re.search(r'\d+', predicted_string)
            predicted_number = int(predicted_number_string.group()) if predicted_number_string else None

            if(predicted_number == test_number):
                correct += 1
        return correct/len(test_files)
    
    def fillLibrary(self, data_folder: str) -> None:
        """
        Fill the library with codebooks from a folder.

        :param data_folder: str. Path to the folder containing the codebooks
        """
        for filename in glob(f'{data_folder}\*.npy'):
            codebook = Codebook()
            codebook.load(filename)
            self.addCodebook(codebook)
    
    def createLibrary(self, audio_folder: str, save_folder: str, N: int = 1024, M: int = 512, n_mfcc: int = 20, size_codebook: int = 32, epsilon: float = 0.01, verbose: bool = False, window: str = 'hamming', beta: float = 14) -> None:
        """
        Create the library from a folder of audio files.

        :param audio_folder: str. Path to the folder containing the audio files
        :param save_folder: str. Path to the folder to save the codebooks
        :param N: int. The number of samples in each frame
        :param M: int. The number of samples to move between frames
        :param n_mfcc: int. The number of MFCC coefficients to return
        :param size_codebook: int. Number of centroids
        :param epsilon: float. Threshold for stopping condition
        :param verbose: bool. Print on each iterations
        :param window: str. The window to apply to the frame. Options are 'hamming', 'hanning', 'blackman', 'bartlett', 'kaiser'
        :param beta: float. The shape parameter for the kaiser window
        """
        if(audio_folder is None or save_folder is None):
            raise ValueError('audio_folder and save_folder must be initialized')

        for filename in glob(f'{audio_folder}\*.wav'):
            name = filename.split('\\')[-1][:-4]
            sampling_rate, audio_data = wavfile.read(filename)

            # Extract features
            mfcc_features = feature_extraction(np.array(audio_data), N, M, sampling_rate, n_mfcc, window, beta)
            
            # Create codebook
            codebook = Codebook(mfcc_features, size_codebook=size_codebook, epsilon=epsilon, verbose=verbose)
            codebook.save(save_folder + "/" + name)
            self.addCodebook(codebook)

    def copy(self):
        """
        Create a copy of the library.

        :return: library: CodeLibrary object
        """
        return CodeLibrary(self.values())
    
    def save(self, folder: str) -> None:
        """
        Save the library to a folder.

        :param folder: str. Path to the folder to save the library
        """
        directory_path = Path(folder)
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        
        for name, codebook in self.items():
            name = name.split('\\')[-1][:-4].split('/')[-1]
            codebook.save(folder + "/" + name)
    
if __name__ == '__main__':
    N = 1024    # Number of samples in each frame
    M = 512     # Number of samples to move between frames
    n_mfcc = 20 # Number of MFCC coefficients to return

    size_codebook = 32  # Number of centroids
    epsilon = 0.01      # Threshold for stopping condition
    verbose = False     # Print on each iterations

    # Create codebooks for the library
    data_folder = Path().resolve() / "Audio Files"
    save_folder = Path().resolve() / "Codebooks"

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
        codebook.save(str(save_folder) + "/" + name + '.npy')
        twelve_codelibrary.addCodebook(codebook)
    
    for filename in zero_train_files:
        name = filename.split('\\')[-1][:-4]
        sampling_rate, audio_data = wavfile.read(filename)

        # Extract features
        mfcc_features = feature_extraction(np.array(audio_data), N, M, sampling_rate, n_mfcc)
        
        # Create codebook
        codebook = Codebook(mfcc_features, size_codebook=size_codebook, epsilon=epsilon, verbose=verbose)
        codebook.save(str(save_folder) + "/" + name + '.npy')
        zero_codelibrary.addCodebook(codebook)

    # Test the library
    print(zero_codelibrary.getClosestCodebookName(feature_extraction(audio_data, N, M, sampling_rate, n_mfcc)))
