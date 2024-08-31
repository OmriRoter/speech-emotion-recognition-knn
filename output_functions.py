import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from processing_functions import load_data, preprocess_data, train_knn, EMOTION_DICT, DATA_DIR
from processing_functions import probability_based_classification, classify_emotions

OUTPUT_DIR = "C:\\Users\\omti9\\OneDrive - Afeka College Of Engineering\\Desktop\\output"


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


def evaluate_model(model: KNeighborsClassifier, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray,
                   y_test: np.ndarray, files_test: list, label_list: list):
    """Evaluate the trained model using various metrics and save the results."""
    y_pred_majority = model.predict(X_test)
    save_classification_report(y_test, y_pred_majority, 'k-NN_majority_voting')
    save_confusion_matrix(y_test, y_pred_majority, 'k-NN_majority_voting', model.classes_)
    export_classification_to_excel(y_test, y_pred_majority, files_test, 'majority_voting')
    hit_rate_majority = calculate_hit_rate(y_test, y_pred_majority)
    print(f"Hit Rate for K-NN model (Majority Voting): {hit_rate_majority:.2f}")

    probabilities = probability_based_classification(X_test, model, label_list)
    y_pred_prob = classify_emotions(probabilities, label_list)
    save_classification_report(y_test, y_pred_prob, 'k-NN_probability_based')
    save_confusion_matrix(y_test, y_pred_prob, 'k-NN_probability_based', label_list)
    export_classification_to_excel(y_test, y_pred_prob, files_test, 'probability_based')
    hit_rate_prob = calculate_hit_rate(y_test, y_pred_prob)
    print(f"Hit Rate for K-NN model (Probability-based): {hit_rate_prob:.2f}")

    # Create performance comparison plots
    create_performance_comparison_plot(y_test, y_pred_majority, y_pred_prob)
    create_accuracy_vs_k_plot(X_train, y_train)
    create_emotion_distribution_plot(y_train)


def main():
    print("Loading data...")
    X, y, file_names = load_data(DATA_DIR)
    print("Data loaded successfully.")

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, files_train, files_test = preprocess_data(X, y, file_names)
    print("Data preprocessed successfully.")

    print("Training KNN model...")
    best_knn = train_knn(X_train, y_train)
    print("Model trained successfully.")

    print("Evaluating model...")
    label_list = list(EMOTION_DICT.values())
    evaluate_model(best_knn, X_train, X_test, y_train, y_test, files_test, label_list)
    print("Model evaluation completed.")


if __name__ == "__main__":
    main()