
import os
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_and_preprocess_data(root_folder):
    # Definirea căilor către folderele de date
    scissors_folder = os.path.join(root_folder, 'scissors')
    rocks_folder = os.path.join(root_folder, 'rock')
    papers_folder = os.path.join(root_folder, 'paper')

    # Funcție pentru a citi și preprocesa imaginile
    def load_and_process_images(folder_path, label):
        images = []
        labels = []
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (100, 100))  # Ajustați dimensiunea imaginilor la nevoie
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertiți imaginea la tonuri de gri
            images.append(img.flatten())  # Aplatizați imaginea
            labels.append(label)
        return images, labels

    # Încărcarea și preprocesarea imaginilor pentru fiecare clasă
    scissors_images, scissors_labels = load_and_process_images(scissors_folder, 'scrissor')
    rocks_images, rocks_labels = load_and_process_images(rocks_folder, 'rock')
    papers_images, papers_labels = load_and_process_images(papers_folder, 'paper')

    # Concatenarea datelor și etichetelor
    all_images = np.concatenate([scissors_images, rocks_images, papers_images], axis=0)
    all_labels = np.concatenate([scissors_labels, rocks_labels, papers_labels], axis=0)

    # Normalizarea datelor
    all_images = all_images / 255.0

    return all_images, all_labels


def train_and_evaluate_model(X_train, y_train, X_test, y_test, algorithm='knn'):
    if algorithm == 'knn':
        model = KNeighborsClassifier(n_neighbors=3)
    elif algorithm == 'nb':
        model = GaussianNB()
    else:
        raise ValueError("Invalid algorithm specified. Choose 'knn' or 'nb'.")

    # Antrenarea modelului
    model.fit(X_train, y_train)

    # Evaluarea modelului
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Afișarea raportului de clasificare
    print(f"Classification Report for {algorithm}:")
    print(classification_report(y_test, predictions))

    # Afișarea matricei de confuzie
    conf_matrix = confusion_matrix(y_test, predictions)
    print(f"Confusion Matrix for {algorithm}:")
    print(conf_matrix)

    return model, accuracy


def main():
    root_folder = './dataset'

    # Încărcarea și preprocesarea datelor
    all_images, all_labels = load_and_preprocess_data(root_folder)

    # Împărțirea datelor în seturi de antrenare și de test
    X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

    # Antrenarea și evaluarea modelului K-Nearest Neighbors
    knn_model, knn_accuracy = train_and_evaluate_model(X_train, y_train, X_test, y_test, algorithm='knn')
    print(f'K-Nearest Neighbors Accuracy: {knn_accuracy}')

    # Antrenarea și evaluarea modelului Naive Bayes
    nb_model, nb_accuracy = train_and_evaluate_model(X_train, y_train, X_test, y_test, algorithm='nb')
    print(f'Naive Bayes Accuracy: {nb_accuracy}')

    # Testarea modelului pe un număr specific de poze aleatoare din setul de test
    num_random_images_to_test = 5  # Modificați la numărul dorit de poze aleatoare pentru testare
    random_test_indices = random.sample(range(len(X_test)), num_random_images_to_test)
    random_test_set = [X_test[i] for i in random_test_indices]
    random_test_labels = [y_test[i] for i in random_test_indices]

    knn_predictions = knn_model.predict(random_test_set)
    nb_predictions = nb_model.predict(random_test_set)

    print("\nRezultatele testării  K-Nearest Neighbors:")
    print_results(random_test_labels, knn_predictions)

    print("\nRezultatele testării Naive Bayes:")
    print_results(random_test_labels, nb_predictions)


def print_results(true_labels, predictions):
    correct_predictions = 0

    for true_label, prediction in zip(true_labels, predictions):
        if true_label != prediction:
            print(f'\033[91mPredictie Gresita:\033[0m Eticheta Reală: {true_label}, Predictie: {prediction}')
        else:
            correct_predictions += 1
            print(f'Predictie Corecta: Eticheta Reală: {true_label}, Predictie: {prediction}')

    accuracy_percentage = (correct_predictions / len(true_labels)) * 100
    print(f'\nProcentaj de Reușită: {accuracy_percentage:.2f}%')


if __name__ == "__main__":
    main()
