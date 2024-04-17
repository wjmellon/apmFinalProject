import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import os
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


import numpy as np
from PIL import Image

# Define random_rotation function
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

# ... the rest of your code ...


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

size = 28

# Function to load and preprocess data
def load_data(image_folders, label_file, img_size=size):
    labels_df = pd.read_csv(label_file)
    labels_df.set_index('image_id', inplace=True)

    # Initialize label encoder
    label_encoder = LabelEncoder()
    all_labels = labels_df['dx'].unique()
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
                image_path = os.path.join(image_folder, file)
                image = Image.open(image_path).resize((img_size, img_size)).convert('RGB')
                image_array = np.array(image) / 255.0
                images.append(image_array.flatten())
                label = labels_df.loc[image_id, 'dx']
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


def load_data_with_augmentation(image_folders, label_file, img_size=size, target_samples=2000):
    images, labels, label_encoder = load_data(image_folders, label_file, img_size)
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
X, Y, label_encoder = load_data_with_augmentation(
    image_folders=image_folders,
    label_file=label_file,
    target_samples=2000  # This is the target number of samples per class
)

num_classes = len(np.unique(Y))
num_features = X.shape[1]
W = np.zeros((num_classes, num_features))
alpha = 0.001
nep = 1000
btch = 256
N = len(Y)
nbtch = ceil(N / btch)
beta1, beta2, epsilon = 0.9, 0.999, 1e-8

losses = []
m, v = np.zeros_like(W), np.zeros_like(W)
K = 1
total_correct_predictions = 0

for epoch in range(nep):
    epoch_loss = 0.0
    correct_predictions = 0
    for i in range(0, N, btch):
        end = min(i + btch, N)
        x_batch = X[i:end]
        y_batch = Y[i:end]
        for j in range(x_batch.shape[0]):
            x_i = x_batch[j][:, np.newaxis]
            y_i = y_batch[j]
            gk = compute_gradient(W, x_i, y_i)
            W -= alpha * gk  # Simple gradient descent update
            scores = np.dot(W, x_i)
            fw = functinonw(scores)
            loss_instance = -np.log(fw[y_i])
            epoch_loss += loss_instance.item()  # Convert numpy array to scalar if needed
            if np.argmax(fw) == y_i:
                correct_predictions += 1
    accuracy = correct_predictions / float(N)
    losses.append(epoch_loss / N)  # Ensure losses are collected for plotting
    print(f'Epoch {epoch + 1}/{nep}, Loss: {epoch_loss / N:.4f}, Accuracy: {accuracy:.4f}')

plt.plot(range(1, nep + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.show()

from collections import defaultdict

# Load test data
test_images = ['./dataverse_files/HAM10000_images_part_1']
x_test, Y_test, test_label_encoder = load_data(test_images, label_file)
test_accuracy = 0
correct_test_predictions = 0

# Initialize dictionaries to track predictions
class_correct = defaultdict(int)
class_total = defaultdict(int)

# Process each test image
for i in range(len(x_test)):
    x_i = x_test[i].reshape(-1, 1)
    scores = np.dot(W, x_i)
    probabilities = functinonw(scores)
    predicted_label_int = np.argmax(probabilities)
    actual_label_int = Y_test[i]

    # Convert numerical labels to original class names for actual and predicted
    actual_label_name = test_label_encoder.inverse_transform([actual_label_int])[0]
    predicted_label_name = test_label_encoder.inverse_transform([predicted_label_int])[0]

    # Print each label and its prediction
    print(f'Image {i + 1}: Actual label - {actual_label_name}, Predicted label - {predicted_label_name}')

    # Update prediction tracking
    class_total[actual_label_int] += 1
    if predicted_label_int == actual_label_int:
        correct_test_predictions += 1
        class_correct[actual_label_int] += 1


# Calculate overall test accuracy
test_accuracy = correct_test_predictions / len(x_test) if len(x_test) > 0 else 0
print(f"Overall Test Accuracy: {test_accuracy:.4f}")

# Calculate and print accuracy for each class
for label_int, total in class_total.items():
    accuracy = class_correct[label_int] / total if total > 0 else 0
    label_name = test_label_encoder.inverse_transform([label_int])[0]
    print(f'Accuracy for {label_name}: {accuracy:.4f}')
