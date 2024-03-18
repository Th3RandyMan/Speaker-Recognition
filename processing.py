from librosa import power_to_db
import numpy as np
from librosa.feature import mfcc, melspectrogram
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, lfilter, freqz
from scipy.io import wavfile
import os

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
    if( audio.shape != (len(audio),) ):
        audio = audio[:,0]

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
            windowed_frame = frame * np.kaiser(N, beta=beta) # zero is rectangle window, 14 is good starting point
            # beta is the shape parameter: hamming is 5, hanning is 6, blackman is 8.6

        speaker_mfccs[i,:] = np.transpose(mfcc(y=windowed_frame, sr=sampling_rate, n_mfcc=n_mfcc, hop_length=N+1))
    return speaker_mfccs

def visualize_mfccs(mfcc_features: np.ndarray, mfcc_x: int, mfcc_y: int, ax=None) -> None:
    """
    Visualize the MFCC features on a scatter plot.

    :param mfcc_features: numpy array of shape (n_frames, n_mfcc)
    :param mfcc_x: int. The index of the MFCC to use for the x-axis
    :param mfcc_y: int. The index of the MFCC to use for the y-axis
    """
    # Extract the MFCCs for the x and y axes
    x = mfcc_features[:, mfcc_x]
    y = mfcc_features[:, mfcc_y]

    if ax is None:
            fig, ax = plt.subplots()

    ax.scatter(x, y, label='Speaker MFCCs')
    ax.set_xlabel(f'MFCC {mfcc_x}')
    ax.set_ylabel(f'MFCC {mfcc_y}')
    ax.set_title('Scatter plot of MFCCs')

    return ax

def apply_notch_filter(input_filepath, output_filepath, notch_freq, Q):
    """
    Apply a notch filter to audio data and save the filtered data.

    :param input_filepath: str. Path to the original audio file
    :param output_filepath: str. Path to save the filtered audio file
    :param notch_freq: float. The frequency to be removed (Hz)
    :param Q: float. The quality factor
    :param filter_type: str. The type of the filter
    """
    # Read the audio file
    sampling_rate, audio_data = wavfile.read(input_filepath)

    # Normalize audio data
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Design notch filter
    b, a = iirnotch(notch_freq / (sampling_rate / 2), Q)

    freq, response = freqz(b, a)

    # Convert frequency to Hz (from rad/sample)
    freq = freq * sampling_rate / (2*np.pi)

    # Plot the frequency response
    plt.figure()
    plt.plot(freq, 20*np.log10(abs(response)), color='blue')
    plt.title('Frequency Response of the Notch Filter')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.grid()
    plt.show()

    # Apply notch filter
    audio_filtered = lfilter(b, a, audio_data)

    # Save filtered audio
    wavfile.write(output_filepath, sampling_rate, np.int16(audio_filtered / np.max(np.abs(audio_filtered)) * 32767))

def apply_notch_filter_to_directory(input_directory, output_directory, notch_freq, Q):
    # Get a list of all .wav files in the input directory
    input_files = [f for f in os.listdir(input_directory) if f.endswith('.wav')]

    # Apply the notch filter to each file
    for input_file in input_files:
        input_filepath = os.path.join(input_directory, input_file)
        
        # Create the output filename by appending "_notched_<freq>Hz" to the original filename
        base_name, ext = os.path.splitext(input_file)
        output_file = f"{base_name}_notched_{notch_freq}Hz{ext}"
        output_filepath = os.path.join(output_directory, output_file)
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        apply_notch_filter(input_filepath, output_filepath, notch_freq, Q)