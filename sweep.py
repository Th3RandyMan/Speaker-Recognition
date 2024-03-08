import pickle 
import numpy as np
from glob import glob
from tqdm import tqdm
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
    zero_test_files = glob(f'{data_folder}\Zero Test\*.wav')

    # Sweeping parameters
    window_list = ['hamming', 'hanning', 'blackman', 'bartlett', 'kaiser']
    beta_list = [*range(2, 15, 2)]
    N_list = [1024, 1440, 2048]
    M_list = [0.4, 0.5, 0.6]
    n_mfcc_list = [20, 40, 60, 80]
    size_codebook_list = [16, 32, 64]

    # Create codelibraries
    twelve_codelibrary = CodeLibrary()
    best_twelve_codebook = CodeLibrary()
    zero_codelibrary = CodeLibrary()
    best_zero_codelibrary = CodeLibrary()

    twelve_best_accuracy = 0.0
    zero_best_accuracy = 0.0
    twelve_acc = defaultdict(np.ndarray)
    zero_acc = defaultdict(np.ndarray)

    # Fill the libraries
    acc_shape = (len(N_list),len(M_list),len(n_mfcc_list),len(size_codebook_list))
    with tqdm(total=np.prod(acc_shape)*(len(window_list) + len(beta_list) - 1)) as pbar:
        for window in window_list:
            beta = [1]
            if window == 'kaiser':
                beta = beta_list
            for b in beta:
                twelve_acc_hold = np.zeros(acc_shape)
                zero_acc_hold = np.zeros(acc_shape)

                for i1, N in enumerate(N_list):
                    for i2, M in enumerate(M_list):
                        for i3, n_mfcc in enumerate(n_mfcc_list):
                            for i4, size_codebook in enumerate(size_codebook_list):
                                twelve_codelibrary.createLibrary(str(data_folder) + "/Twelve Train", str(save_folder), N=N, M=int(N*M), n_mfcc=n_mfcc, size_codebook=size_codebook, window=window, beta=b)
                                zero_codelibrary.createLibrary(str(data_folder) + "/Zero Train", str(save_folder), N=N, M=int(N*M), n_mfcc=n_mfcc, size_codebook=size_codebook, window=window, beta=b)

                                # Predict the test files
                                twelve_accuracy = twelve_codelibrary.getAccuracy(twelve_test_files)
                                twelve_acc_hold[i1,i2,i3,i4] = twelve_accuracy
                                if(twelve_accuracy > twelve_best_accuracy):
                                    best_twelve_codebook = twelve_codelibrary.copy()
                                    twelve_best_accuracy = twelve_accuracy
                                
                                zero_accuracy = zero_codelibrary.getAccuracy(zero_test_files)
                                zero_acc_hold[i1,i2,i3,i4] = zero_accuracy
                                if(zero_accuracy > zero_best_accuracy):
                                    best_zero_codelibrary = zero_codelibrary.copy()
                                    zero_best_accuracy = zero_accuracy
                                
                                pbar.update(1)

                if(window == 'kaiser'):
                    twelve_acc[window+b] = twelve_acc_hold
                    zero_acc[window+b] = zero_acc_hold
                else:
                    twelve_acc[window] = twelve_acc_hold
                    zero_acc[window] = zero_acc_hold

    with open('Codebooks/twelve_acc.pkl', 'wb') as f:
        pickle.dump(twelve_acc, f)
    with open('Codebooks/zero_acc.pkl', 'wb') as f:
        pickle.dump(zero_acc, f)
    
    best_twelve_codebook.save("Codebooks/Best Twelve")
    best_zero_codelibrary.save("Codebooks/Best Zero")
    print(f"Twelve Best Accuracy: {twelve_best_accuracy}")
    print(f"Zero Best Accuracy: {zero_best_accuracy}")

