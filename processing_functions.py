import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import opensmile

# Set directory path to data and configuration parameters for the model
DATA_DIR = "C:\\Users\\omti9\\OneDrive - Afeka College Of Engineering\\Desktop\\EmoDB_1"
TEST_SIZE = 0.2  # Fraction of data to be used as test set
RANDOM_STATE = 1  # Seed for random operations to ensure reproducibility
MIN_NEIGHBORS = 1  # Minimum number of neighbors for tuning KNN
MAX_NEIGHBORS = 50  # Maximum number of neighbors for tuning KNN
CV_FOLDS = 8  # Number of cross-validation folds


# Dictionary mapping emotion codes to human-readable strings
EMOTION_DICT = {
    'A': 'Angst (fear)',
    'E': 'Ekel (disgust)',
    'F': 'Freude (happiness)',
    'L': 'Langeweile (boredom)',
    'N': 'Neutral',
    'T': 'Trauer (sadness)',
    'W': 'Wut (anger)'
}

# Create an OpenSmile object for feature extraction using specific settings
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)


def extract_features(file_path: str) -> np.ndarray:
    """Extract features from a single audio file using OpenSmile."""
    features = smile.process_file(file_path)
    return features.iloc[0].to_numpy()


def label_from_filename(file_name: str) -> str:
    """Extract emotion label from a filename using the last two characters before the extension."""
    emotion_code = file_name.split('.')[0][-2]
    return EMOTION_DICT.get(emotion_code, 'Unknown')


def load_data(data_dir: str) -> tuple:
    """Load data from directory, extract features and labels for each audio file."""
    data = []
    labels = []
    file_names = []
    for file_path in glob.glob(os.path.join(data_dir, '*.wav')):
        file_name = os.path.basename(file_path)
        features = extract_features(file_path)
        data.append(features)
        labels.append(label_from_filename(file_name))
        file_names.append(file_name)
    return np.array(data), np.array(labels), file_names


def preprocess_data(X: np.ndarray, y: np.ndarray, file_names: list) -> tuple:
    """Split and standardize the dataset for training and testing."""
    X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
        X, y, file_names, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, files_train, files_test


def train_knn(X_train: np.ndarray, y_train: np.ndarray) -> KNeighborsClassifier:
    """Train a K-Nearest Neighbors classifier using a grid search over possible neighbor counts."""
    param_grid = {'n_neighbors': range(MIN_NEIGHBORS, MAX_NEIGHBORS)}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=CV_FOLDS)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def probability_based_classification(X: np.ndarray, model: KNeighborsClassifier, label_list: list) -> list:
    """Calculate classification probabilities based on nearest neighbors."""
    k = model.n_neighbors
    probabilities = []
    for x in X:
        distances, indices = model.kneighbors([x])
        emotion_counts = {label: 0 for label in label_list}
        for index in indices[0]:
            label = model._y[index]
            emotion_counts[label_list[label]] += 1
        emotion_probabilities = {}
        for label in label_list:
            Vij = emotion_counts[label]
            emotion_probabilities[label] = (Vij * k) / (k ** 2)
        probabilities.append(emotion_probabilities)
    return probabilities


def classify_emotions(probabilities: list, label_list: list) -> np.ndarray:
    """Determine the most probable emotion from calculated probabilities."""
    predicted_emotions = []
    for prob in probabilities:
        predicted_emotion = max(prob, key=prob.get)
        predicted_emotions.append(predicted_emotion)
    return np.array(predicted_emotions)
