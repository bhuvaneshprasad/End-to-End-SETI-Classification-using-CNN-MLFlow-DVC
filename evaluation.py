from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

def evaluate(dataset_path, model):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_path,
            shuffle=True,
            image_size = (256, 256),
            batch_size = 32
        )
    
    dataset = dataset.shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    class_labels = ['brightpixel','narrowband',
                 'narrowbanddrd','noise',
                 'squarepulsednarrowband','squiggle',
                 'squigglesquarepulsednarrowband']
    
    # Get true labels and predictions
    y_true = []
    y_pred = []

    # Iterate over the test dataset to get predictions
    for images, labels in dataset.as_numpy_iterator():
        y_true.extend(labels)
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)

    # Print metrics
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)
    
    # Plot confusion matrix
    plt.figure(figsize=(4, 3))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', 
                xticklabels=class_labels, 
                yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    model = tf.keras.models.load_model(Path('artifacts\\model_training\\model.keras'))
    evaluate(Path('artifacts\\data_ingestion\\seti-sub\\seti_signals_sub\\test'), model)

if __name__ == '__main__':
    main()