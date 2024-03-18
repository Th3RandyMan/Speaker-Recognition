# Speaker Recognition
EEC 201 Final Project

Created by Randall Fowler and Conor King

## Overview
Speaker recognition refers to the classification of which person is speaking in an audio recording. This project uses Mel-Frequency Cepstrum Coefficients as features for a clustering algorithm. Each cluster will have a centroid to represent the center of the cluster, and each centroid is saved into a codebook.

For ease of use, many of the functions for this project have been implemented as class methods. The `codelibrary` will create, store, and reference codebooks created from training data. Runtime for the algorithm is relatively short as it may only take a minute or two to complete training or calculating accuracy of the model.

## Requirements
* numpy
* librosa
* matplotlib
* scipy
* tqdm

## Usuage
All audio data used for the project are stored within the `Audio Files` folder and codebooks will be stored in the `Codebooks` folder. All other necessary files can be found within the root director for the project.

To get a better understanding of the implementation of the project, you can take a look at the `Preview.ipynb`. This file was used as a starting point to create the processing method and start the tests.

To use the project, a good place to start would be Test 9 in the `Tests.ipynb` file. It shows how a `codelibrary` can be created. This is a collection of codebooks, and each codebook is created for individual training audio files in the specified folder.

If you wish to optimize on hyperparameters, there is a `sweep.py` that can be used to select different ranges of hyperparameters to search to find an optimal parameter set. The results of the sweep will be stored in the `Codebooks` folder as a pickle file. An example of viewing the result can be seen in the `Preview.ipynb` file.


## Results
Accuracy of different tests can be found in the `Tests.ipynb` file. To read about the project, there is a pdf file within the `Paper` folder. To watch a video relating to the project, you can find a mp4 file within the `Video` folder.

## Acknowledgements
This project was completed for EEC 201 at UC Davis, Winter 2024, by Randall Fowler and Conor King.
