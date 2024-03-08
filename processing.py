from librosa import power_to_db
import numpy as np
from librosa.feature import mfcc, melspectrogram

# Could apply filtering or other preprocessing before calling function
def feature_extraction(audio: np.ndarray, N: int, M: int, sampling_rate: int, n_mfcc: int, window: str = 'hamming', beta: float = 14) -> np.ndarray:
    """
    Perform frame blocking and extract the MFCC features from each frame of the audio signal.

    :param audio: numpy array of shape (n_samples,)
    :param N: int. The number of samples in each frame
    :param M: int. The number of samples to move between frames
    :param sampling_rate: int. The sampling rate of the audio signal
    :param n_mfcc: int. The number of MFCC coefficients to return
    :param window: str. The window to apply to the frame. Options are 'hamming', 'hanning', 'blackman', 'bartlett', 'kaiser'
    :param beta: float. The shape parameter for the kaiser window

    :return: mfcc_features: numpy array of shape (n_frames, n_mfcc)
    """
    speaker_mfccs = np.zeros((1 + (len(audio) - N + 1)//M, n_mfcc)) # Initialize array to hold MFCCs

    for i, sample in enumerate(range(0, len(audio) - N + 1, M)):
        frame = audio[sample:sample+N]
        
        # Add in optional filters
        if window == 'hamming':
            windowed_frame = frame * np.hamming(N)  # Hamming window
        elif window == 'hanning':
            windowed_frame = frame * np.hanning(N)
        elif window == 'blackman':
            windowed_frame = frame * np.blackman(N)
        elif window == 'bartlett':
            windowed_frame = frame * np.bartlett(N)
        elif window == 'kaiser':
            windowed_frame = frame * np.kaiser(N, beta=14) # zero is rectangle window, 14 is good starting point
            # beta is the shape parameter: hamming is 5, hanning is 6, blackman is 8.6

        speaker_mfccs[i,:] = np.transpose(mfcc(y=windowed_frame, sr=sampling_rate, n_mfcc=n_mfcc, hop_length=N+1))
    return speaker_mfccs