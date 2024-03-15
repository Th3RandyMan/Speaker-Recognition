from glob import glob
from pathlib import Path
from codelibrary import CodeLibrary

data_folder = Path().resolve() / "Audio Files"

# Gather the test and train files
twelve_test_files = glob(f'{data_folder}\Twelve Test\*.wav')
twelve_train_files = glob(f'{data_folder}\Twelve Train\*.wav')
zero_test_files = glob(f'{data_folder}\Zero Test\*.wav')
zero_train_files = glob(f'{data_folder}\Zero Train\*.wav')

twelve_zero_codelibrary = CodeLibrary()
twelve_zero_codelibrary.fillLibrary("Codebooks/Twelve Zero")
#twelve_zero_train_files = twelve_train_files + zero_train_files
twelve_zero_test_files = twelve_test_files + zero_test_files

# for filename in twelve_zero_test_files:
#     name = filename.split('\\')[-1][:-4]
#     print(f'Twelve Zero Train: {name}')
#     print(f'Predicted: {twelve_zero_codelibrary.predict(filename,N = 1024, M = 409, n_mfcc = 40)}')
#     print()

print(f"Accuracy: {twelve_zero_codelibrary.getAccuracy(twelve_zero_test_files,N = 1024, M = 409, n_mfcc = 40)}")

# # Create codelibraries
# twelve_codelibrary = CodeLibrary()
# twelve_codelibrary.fillLibrary("Codebooks/Best Twelve")
# zero_codelibrary = CodeLibrary()
# zero_codelibrary.fillLibrary("Codebooks/Best Zero")

# # Predict the test files
# for filename in twelve_test_files:
#     name = filename.split('\\')[-1][:-4]
#     print(f'Twelve Test: {name}')
#     print(f'Predicted: {twelve_codelibrary.predict(filename)}')
#     print()

# for filename in zero_test_files:
#     name = filename.split('\\')[-1][:-4]
#     print(f'Zero Test: {name}')
#     print(f'Predicted: {zero_codelibrary.predict(filename)}')
#     print()