# Emotion Recognition from Speech

## Overview

This project implements a machine learning system for recognizing emotions from speech audio files. It uses the EmoDB (Berlin Database of Emotional Speech) dataset and applies a K-Nearest Neighbors (KNN) classifier to predict emotions based on acoustic features extracted from the audio.

## Features

- Audio feature extraction using OpenSMILE
- K-Nearest Neighbors classification with hyperparameter tuning
- Two classification strategies: majority voting and probability-based
- Comprehensive evaluation metrics including classification reports and confusion matrices
- Visualization of results and model performance

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- OpenSMILE (Python wrapper)

## Setup

1. Clone the repository:
   ```
   git clone [repository-url]
   cd emotion-recognition-speech
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the EmoDB dataset and place the audio files in a directory of your choice.

4. Ensure OpenSMILE is properly installed and configured.

5. Create an output directory where the results will be saved.

6. Update the path variables in the code:
   - Open `processing_functions.py` and update the `DATA_DIR` variable with the path to your EmoDB dataset directory.
   - Open `output_functions.py` and update the `OUTPUT_DIR` variable with the path to your desired output directory.

## Directory Structure

Ensure your project has the following directory structure:

```
emotion-recognition-speech/
│
├── main.py
├── processing_functions.py
├── output_functions.py
├── README.md
│
├── data/
│   └── [EmoDB audio files]
│
└── output/
    └── [Results will be saved here]
```

## Usage

1. Prepare your environment:
   - Ensure you have set up the correct paths in `processing_functions.py` and `output_functions.py` as mentioned in the Setup section.
   - Make sure your EmoDB dataset is in the directory specified by `DATA_DIR`.
   - Ensure the output directory specified by `OUTPUT_DIR` exists.

2. Run the main script to process the data, train the model, and generate results:
   ```
   python main.py
   ```

3. The script will execute the entire pipeline:
   - Load and preprocess the audio data
   - Extract features using OpenSMILE
   - Train the KNN model
   - Perform classifications using both majority voting and probability-based methods
   - Generate evaluation metrics and visualizations

4. Check the output directory for results:
   - Classification reports (CSV files)
   - Confusion matrices (PNG files)
   - Performance comparison plots (PNG files)
   - Accuracy vs. k plot (PNG file)
   - Emotion distribution plot (PNG file)

## Customization

- To adjust the test set size, modify the `TEST_SIZE` variable in `processing_functions.py`.
- To change the range of K values for the KNN classifier, modify `MIN_NEIGHBORS` and `MAX_NEIGHBORS` in `processing_functions.py`.
- To adjust the number of cross-validation folds, modify `CV_FOLDS` in `processing_functions.py`.

## Troubleshooting

- If you encounter path-related errors, double-check that you've correctly set `DATA_DIR` and `OUTPUT_DIR` in the respective files.
- Ensure that the EmoDB audio files are directly in the `DATA_DIR` directory, not in subdirectories.
- If you face issues with OpenSMILE, make sure it's correctly installed and its Python wrapper is properly configured.

## Key Components

### Data Processing

- Audio features are extracted using OpenSMILE's eGeMAPSv02 feature set.
- The dataset is split into training and testing sets, with stratification to ensure balanced emotion distribution.
- Features are standardized using Scikit-learn's StandardScaler.

### Model Training

- A K-Nearest Neighbors classifier is trained using GridSearchCV for hyperparameter tuning.
- The optimal number of neighbors is determined through cross-validation.

### Classification Strategies

1. **Majority Voting**: The standard KNN classification approach.
2. **Probability-Based**: A custom implementation that calculates emotion probabilities based on the k-nearest neighbors.

### Evaluation

- Classification reports and confusion matrices are generated for both classification strategies.
- Additional visualizations include:
  - Performance comparison between majority voting and probability-based methods
  - Accuracy vs. k plot to visualize the impact of the number of neighbors
  - Emotion distribution in the dataset

## Output

Results and visualizations are saved in the directory specified by `OUTPUT_DIR`, including:

- Classification reports (CSV)
- Confusion matrices (PNG)
- Performance comparison plots (PNG)
- Accuracy vs. k plot (PNG)
- Emotion distribution plot (PNG)

## Note on Reproducibility

The random state is set to ensure reproducibility of results. However, slight variations may occur due to differences in system configurations or library versions.

## Future Improvements

- Implement additional classification algorithms for comparison (e.g., SVM, Random Forest)
- Explore deep learning approaches such as Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs)
- Incorporate real-time audio processing for live emotion recognition

## Contributors

- Omri Roter

## Contact

For questions, suggestions, or contributions, please contact:

Omri Roter - Omri99Roter@gmail.com


## Acknowledgments
- The EmoDB dataset providers
- OpenSMILE developers for the feature extraction toolkit
