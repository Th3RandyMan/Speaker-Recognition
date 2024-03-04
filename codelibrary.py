import numpy as np
from glob import glob
from pathlib import Path
from codebook import Codebook

class CodeLibrary:
    def __init__(self, codebooks=None):
        """
        Initialize the CodeLibrary with a dictionary of codebooks.

        :param codebooks: dictionary of Codebook objects
        """
        self.codebooks = codebooks

    def __str__(self) -> str:
        return f'CodeLibrary: {self.codebooks}'

    def addCodebook(self, codebook: Codebook) -> None:
        """
        Add a codebook to the library.

        :param codebook: Codebook object
        """
        self.codebooks[codebook.name] = codebook

    def removeCodebook(self, codebook: Codebook) -> None:
        """
        Remove a codebook from the library.

        :param codebook: Codebook object
        """
        del self.codebooks[codebook.name]

    def getCodebook(self, name: str) -> Codebook:
        """
        Get a codebook from the library.

        :param name: name of the codebook
        :return: codebook: Codebook object
        """
        return self.codebooks[name]

    def getDistance(self, data: np.ndarray) -> list[float]:
        """
        Get the distance of the data to each codebook in the library.

        :param data: numpy array of shape (n_samples, n_dimensions)
        :return: distances: list of floats
        """
        distances = []
        for codebook in self.codebooks.values():
            distances.append(codebook.getDistance(data))
        return distances

    def getClosestCodebook(self, data: np.ndarray) -> Codebook:
        """
        Get the closest codebook to the data in the library.

        :param data: numpy array of shape (n_samples, n_dimensions)
        :return: codebook: Codebook object
        """
        distances = self.getDistance(data)
        closest_codebook_name = min(self.codebooks, key=lambda x: distances[self.codebooks[x]])
        return self.codebooks[closest_codebook_name]
    
if __name__ == '__main__':
    # Create codebooks for the library
    data_folder = Path().resolve() / "Audio Files"

    twelve_train_files = glob(f'{data_folder}\Twelve Train\*.wav')
    zero_train_files = glob(f'{data_folder}\Zero Train\*.wav')

    # Create codebooks
