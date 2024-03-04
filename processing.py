import numpy as np
from librosa.feature import mfcc 


# Could apply filtering or other preprocessing before calling function
def feature_extraction(audio: np.ndarray, N: int, M: int, sampling_rate: int, n_mfcc: int) -> np.ndarray:
    """
    Perform frame blocking and extract the MFCC features from each frame of the audio signal.

    :param audio: numpy array of shape (n_samples,)
    :param N: int. The number of samples in each frame
    :param M: int. The number of samples to move between frames
    :param sampling_rate: int. The sampling rate of the audio signal
    :param n_mfcc: int. The number of MFCC coefficients to return
    :return: mfcc_features: numpy array of shape (n_frames, n_mfcc)
    """
    speaker_mfccs = np.zeros((1 + (len(audio) - N + 1)//M, n_mfcc)) # Initialize array to hold MFCCs

    for i, sample in enumerate(range(0, len(audio) - N + 1, M)):
        frame = audio[sample:sample+N]
        
        # Add in optional filters
        windowed_frame = frame * np.hamming(N)  # Hamming window

        speaker_mfccs[i,:] = np.transpose(mfcc(y=windowed_frame, sr=sampling_rate, n_mfcc=n_mfcc))
    return speaker_mfccs