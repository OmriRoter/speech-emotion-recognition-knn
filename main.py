import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import opensmile

# Define constants for file paths and configuration parameters
DATA_DIR = "C:\\Users\\omti9\\OneDrive - Afeka College Of Engineering\\Desktop\\לימודים\\אפקה\\שנה ב\\סימסטר א\\עקרונות עיבוד דיבור\\EmoDB_1"
OUTPUT_DIR = "C:\\Users\\omti9\\OneDrive - Afeka College Of Engineering\\Desktop\\output"
TEST_SIZE = 0.2
RANDOM_STATE = 1
MIN_NEIGHBORS = 1
MAX_NEIGHBORS = 50
CV_FOLDS = 8

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


def evaluate_model(model: KNeighborsClassifier, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, files_test: list, label_list: list):
    """Evaluate the trained model using various metrics and save the results."""
    y_pred_majority = model.predict(X_test)
    save_classification_report(y_test, y_pred_majority, 'k-NN_majority_voting')
    save_confusion_matrix(y_test, y_pred_majority, 'k-NN_majority_voting', model.classes_)
    export_classification_to_excel(y_test, y_pred_majority, files_test, 'majority_voting')
    hit_rate_majority = calculate_hit_rate(y_test, y_pred_majority)
    print(f"Hit Rate for K-NN model (Majority Voting): {hit_rate_majority:.2f}")
    create_hit_rate_excel(files_test, y_test, y_pred_majority, 'majority_voting')

    probabilities = probability_based_classification(X_test, model, label_list)
    y_pred_prob = classify_emotions(probabilities, label_list)
    save_classification_report(y_test, y_pred_prob, 'k-NN_probability_based')
    save_confusion_matrix(y_test, y_pred_prob, 'k-NN_probability_based', label_list)
    export_classification_to_excel(y_test, y_pred_prob, files_test, 'probability_based')
    hit_rate_prob = calculate_hit_rate(y_test, y_pred_prob)
    print(f"Hit Rate for K-NN model (Probability-based): {hit_rate_prob:.2f}")
    create_hit_rate_excel(files_test, y_test, y_pred_prob, 'probability_based')

    # Create performance comparison plots
    create_performance_comparison_plot(y_test, y_pred_majority, y_pred_prob)
    create_accuracy_vs_k_plot(X_train, y_train)
    create_emotion_distribution_plot(y_train)


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(OUTPUT_DIR, f'{model_name}_classification_report.csv'), index=True)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, labels: list):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name}_confusion_matrix.png'), bbox_inches='tight')
    plt.close()


def export_classification_to_excel(y_true: np.ndarray, y_pred: np.ndarray, file_names: list, method: str):
    df = pd.DataFrame({
        'Recording Name': file_names,
        'True Emotion': y_true,
        'Predicted Emotion': y_pred,
        'Is Accurate': y_true == y_pred
    })
    df['Is Accurate'] = df['Is Accurate'].apply(lambda x: 'Yes' if x else 'No')
    csv_path = os.path.join(OUTPUT_DIR, f"{method}_classification_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"{method.capitalize()} classification report saved to {csv_path}")


def calculate_hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    hit_rate = correct_predictions / total_predictions
    return hit_rate


def create_hit_rate_excel(file_names: list, y_true: np.ndarray, y_pred: np.ndarray, method: str):
    df = pd.DataFrame({
        'Recording Name': file_names,
        'True Emotion': y_true,
        'Predicted Emotion': y_pred,
        'Is Accurate': y_true == y_pred
    })
    df['Is Accurate'] = df['Is Accurate'].apply(lambda x: 'Yes' if x else 'No')
    csv_path = os.path.join(OUTPUT_DIR, f"{method}_hit_rate_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"{method.capitalize()} Hit Rate CSV report saved to {csv_path}")


def create_performance_comparison_plot(y_true: np.ndarray, y_pred_majority: np.ndarray, y_pred_prob: np.ndarray):
    """Create a bar plot comparing the F1 scores of majority voting and probability-based methods."""
    f1_majority = classification_report(y_true, y_pred_majority, output_dict=True)['weighted avg']['f1-score']
    f1_prob = classification_report(y_true, y_pred_prob, output_dict=True)['weighted avg']['f1-score']

    plt.figure(figsize=(8, 6))
    plt.bar(['Majority Voting', 'Probability-based'], [f1_majority, f1_prob])
    plt.ylabel('F1 Score')
    plt.title('Performance Comparison of Decision Strategies')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'performance_comparison.png'), bbox_inches='tight')
    plt.close()


def create_accuracy_vs_k_plot(X_train: np.ndarray, y_train: np.ndarray):
    """Create a line plot showing the relationship between accuracy and the number of neighbors (k)."""
    k_values = range(1, 31)
    accuracies = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        accuracies.append(knn.score(X_train, y_train))

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. k')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_vs_k.png'), bbox_inches='tight')
    plt.close()


def create_emotion_distribution_plot(labels: np.ndarray):
    """Create a histogram showing the distribution of emotions in the dataset."""
    emotion_counts = pd.Series(labels).value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.title('Distribution of Emotions in the Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'emotion_distribution.png'), bbox_inches='tight')
    plt.close()


def main():
    """Main function to load data, process it, train a model, and evaluate it."""
    X, y, file_names = load_data(DATA_DIR)
    X_train, X_test, y_train, y_test, files_train, files_test = preprocess_data(X, y, file_names)
    best_knn = train_knn(X_train, y_train)
    label_list = list(EMOTION_DICT.values())
    evaluate_model(best_knn, X_train, X_test, y_train, y_test, files_test, label_list)


if __name__ == "__main__":
    main()