import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import os
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


def random_rotation(image_array, max_angle=90):
    """
    Rotate the image by a random angle between 0 and max_angle degrees.
    image_array: a flattened image array.
    max_angle: maximum rotation angle in degrees.
    """
    image = Image.fromarray(image_array.reshape(size, size, 3).astype('uint8'), 'RGB')
    random_angle = np.random.randint(-max_angle, max_angle)
    return np.array(image.rotate(random_angle)).flatten()

# Define random_flip function
def random_flip(image_array):
    """
    Flip the image randomly horizontally or vertically.
    image_array: a flattened image array.
    """
    image = Image.fromarray(image_array.reshape(size, size, 3).astype('uint8'), 'RGB')
    if np.random.rand() > 0.5:
        return np.array(image.transpose(Image.FLIP_LEFT_RIGHT)).flatten()
    else:
        return np.array(image.transpose(Image.FLIP_TOP_BOTTOM)).flatten()



# Function to compute softmax probabilities
def functinonw(z):
    ez = np.exp(z - np.max(z, axis=0, keepdims=True))
    return ez / np.sum(ez, axis=0, keepdims=True)

# Function to compute gradient for updating weights
def compute_gradient(W, x_i, y_i):
    scores = np.dot(W, x_i)
    probabilities = functinonw(scores)
    probabilities[y_i] -= 1
    grad = np.outer(probabilities, x_i)
    return grad

# Function to load and preprocess data
size = 32

# Function to load and preprocess data
# Function to load and preprocess data
def load_data(image_folders, label_file, img_size=size, specific_labels=None, column = 'dx'):
    labels_df = pd.read_csv(label_file)
    labels_df.set_index('image_id', inplace=True)

    label_encoder = LabelEncoder()

    if specific_labels:
        all_labels = specific_labels
    else:
        all_labels = labels_df[column].unique()

    label_encoder.fit(all_labels)

    images = []
    label_list = []
    missing_labels = []

    # Process each image folder
    for image_folder in image_folders:
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        for file in image_files:
            image_id = file.split('.')[0]
            if image_id in labels_df.index:
                label = labels_df.loc[image_id, column]
                if specific_labels and label not in specific_labels:
                    continue  # Skip images with labels not in specific_labels
                image_path = os.path.join(image_folder, file)
                image = Image.open(image_path).resize((img_size, img_size)).convert('RGB')
                image_array = np.array(image) / 255.0
                images.append(image_array.flatten())
                label_list.append(label)
            else:
                missing_labels.append(file)

    if missing_labels:
        print("Warning: Missing labels for files:", missing_labels)
    if not images:
        print("Error: No images were loaded.")
        return None, None, None

    # Encode labels
    labels = label_encoder.transform(label_list)
    return np.array(images), labels, label_encoder


def load_data_with_augmentation(image_folders, label_file, img_size=size, target_samples=2000, specific_labels=None,column = 'dx'):
    images, labels, label_encoder = load_data(image_folders, label_file, img_size, specific_labels=specific_labels, column = column)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_counts_dict = dict(zip(unique_labels, label_counts))

    # Determine how many samples each class needs
    augmentation_counts = {label: max(0, target_samples - count) for label, count in label_counts_dict.items()}

    augmented_images = []
    augmented_labels = []

    for image, label in zip(images, labels):
        num_augmentations = augmentation_counts[label]

        # Perform the necessary number of augmentations for this label
        for _ in range(num_augmentations):
            aug_image = random_rotation(image)
            augmented_images.append(aug_image)
            augmented_labels.append(label)

            # You could add more types of augmentation here
            # and reduce num_augmentations accordingly

            # Update the counts and break if we reach the target
            augmentation_counts[label] -= 1
            if augmentation_counts[label] <= 0:
                break

    # Combine original and augmented images
    images = np.vstack((images, np.array(augmented_images)))
    labels = np.hstack((labels, np.array(augmented_labels)))

    # Shuffle the dataset
    images, labels = shuffle(images, labels, random_state=0)

    return images, labels, label_encoder

image_folders = ['./dataverse_files/HAM10000_images_part_1', './dataverse_files/HAM10000_images_part_2']
label_file = './dataverse_files/HAM10000_metadata'
X, Y, label_encoder = load_data(
    image_folders=image_folders,
    label_file=label_file,
)

num_classes = len(np.unique(Y))
num_features = X.shape[1]
W = np.zeros((num_classes, num_features))
nep = 2000
btch = 256
N = len(Y)
nbtch = ceil(N / btch)
beta1, beta2, epsilon = 0.9, 0.999, 1e-8

losses = []
accuracies = []
m, v = np.zeros_like(W), np.zeros_like(W)
K = 1
total_correct_predictions = 0

initial_lr = 0.001  # Initial learning rate
decay_rate = 0.99  # Learning rate decay rate
decay_steps = 20  # Decay steps

for epoch in range(nep):
    alpha = initial_lr * (decay_rate ** (epoch // decay_steps))
    epoch_loss = 0.0
    correct_predictions = 0
    for i in range(int(nbtch)):
        batch_start = i * btch
        batch_end = min(batch_start + btch, N)
        for j in range(batch_start, batch_end):
            x_i = X[j].reshape(-1, 1)
            y_i = Y[j]
            gk = compute_gradient(W, x_i, y_i)
            m = beta1 * m + (1 - beta1) * gk
            v = beta2 * v + (1 - beta2) * gk * gk
            m_hat = m / (1 - beta1 ** K)
            v_hat = v / (1 - beta2 ** K)
            W -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)
            scores = np.dot(W, x_i)
            fw = functinonw(scores)
            loss = -np.log(fw[y_i])
            epoch_loss += loss
            predicted = np.argmax(fw)
            if predicted == y_i:
                correct_predictions += 1
            K += 1
    accuracy = correct_predictions / N
    losses.append(epoch_loss / N)
    average_loss = epoch_loss / N
    accuracies.append(accuracy)

    # convert debug
    if isinstance(average_loss, np.ndarray) and average_loss.size == 1:
        average_loss = average_loss.item()

    total_correct_predictions += correct_predictions
    print(f'Epoch {epoch + 1}/{nep}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')

print(f"Final Training Accuracy: {total_correct_predictions / (N * nep):.4f}")

plt.plot(range(1, nep + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.show()

plt.plot(range(1, nep + 1), accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.show()

from collections import defaultdict

# Initialize dictionaries to track correct predictions and total counts for each class
correct_predictions_per_class = defaultdict(int)
total_predictions_per_class = defaultdict(int)

# Load test data
test_images = ['./dataverse_files/ISIC2018_Task3_Test_Images']
label_file = './dataverse_files/ISIC2018_Task3_Test_GroundTruth.csv'
x_test, Y_test, test_label_encoder = load_data(test_images, label_file)

# Loop through each test instance
for i in range(len(x_test)):
    x_i = x_test[i].reshape(-1, 1)
    scores = np.dot(W, x_i)
    probabilities = functinonw(scores)
    predicted_label_int = np.argmax(probabilities)
    actual_label_int = Y_test[i]

    # Convert integer labels back to original class labels
    predicted_label = test_label_encoder.inverse_transform([predicted_label_int])[0]
    actual_label = test_label_encoder.inverse_transform([actual_label_int])[0]

    # Update counts in dictionaries
    total_predictions_per_class[actual_label] += 1
    if predicted_label == actual_label:
        correct_predictions_per_class[actual_label] += 1

# Calculate and print overall test accuracy
total_correct_predictions = sum(correct_predictions_per_class.values())
total_test_samples = len(x_test)
test_accuracy = total_correct_predictions / total_test_samples if total_test_samples > 0 else 0
print(f"Overall Test Accuracy: {test_accuracy:.4f}")

# Calculate and print accuracy for each class
for label, total_count in total_predictions_per_class.items():
    accuracy = correct_predictions_per_class[label] / total_count if total_count > 0 else 0
    print(f"Accuracy for {label}: {accuracy:.4f}")
