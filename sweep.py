import numpy as np
from glob import glob
from pathlib import Path
from scipy.io import wavfile
from codebook import Codebook
from codelibrary import CodeLibrary
from collections import defaultdict
from processing import feature_extraction



if __name__ == '__main__':
    # Path to the data folder
    data_folder = Path().resolve() / "Audio Files"
    save_folder = Path().resolve() / "Codebooks"

    # Gather the test and train files
    twelve_test_files = glob(f'{data_folder}\Twelve Test\*.wav')
    #twelve_train_files = glob(f'{data_folder}\Twelve Train\*.wav')
    zero_test_files = glob(f'{data_folder}\Zero Test\*.wav')
    #zero_train_files = glob(f'{data_folder}\Zero Train\*.wav')

    # Create codelibraries
    twelve_codelibrary = CodeLibrary()
    best_twelve_codebook = CodeLibrary()
    zero_codelibrary = CodeLibrary()
    best_zero_codelibrary = CodeLibrary()

    # Fill the libraries
    for window in ['hamming', 'hanning', 'blackman', 'bartlett', 'kaiser']:
        beta = [*range(1, 15, 2)]
        if window == 'kaiser':
            beta = [1]
        for b in beta:
            for N in [1024, 1440, 2048, 4096]:
                for M in [0.4, 0.5, 0.6, 0.7]:
                    for n_mfcc in [20, 40, 60, 80]:
                        for size_codebook in [32, 64, 128, 256]:
                            twelve_codelibrary.createLibrary(str(data_folder) + "/Twelve Train", str(save_folder), N=N, M=M, n_mfcc=n_mfcc, size_codebook=size_codebook, window=window, beta=b)
                            zero_codelibrary.createLibrary(str(data_folder) + "/Zero Train", str(save_folder), N=N, M=M, n_mfcc=n_mfcc, size_codebook=size_codebook, window=window, beta=b)



                            
                            


    # Test the library
    print(zero_codelibrary.getClosestCodebookName(feature_extraction(audio_data, N, M, sampling_rate, n_mfcc)))
