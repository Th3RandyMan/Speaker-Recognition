from glob import glob
from pathlib import Path
from codelibrary import CodeLibrary

data_folder = Path().resolve() / "Audio Files"

twelve_test_files = glob(f'{data_folder}\Twelve Test\*.wav')
twelve_train_files = glob(f'{data_folder}\Twelve Train\*.wav')
zero_test_files = glob(f'{data_folder}\Zero Test\*.wav')
zero_train_files = glob(f'{data_folder}\Zero Train\*.wav')

# Create codelibraries
twelve_codelibrary = CodeLibrary()
twelve_codelibrary.fillLibrary("Codebooks/Twelve")
zero_codelibrary = CodeLibrary()
zero_codelibrary.fillLibrary("Codebooks/Zero")

for filename in twelve_test_files:
    name = filename.split('\\')[-1][:-4]
    print(f'Twelve Test: {name}')
    print(f'Predicted: {twelve_codelibrary.predict(filename)}')
    print()

for filename in zero_test_files:
    name = filename.split('\\')[-1][:-4]
    print(f'Zero Test: {name}')
    print(f'Predicted: {zero_codelibrary.predict(filename)}')
    print()