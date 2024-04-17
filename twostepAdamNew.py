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
# Function to load and preprocess data
def load_data(image_folders, label_file, img_size=size, specific_labels=None, column = 'dx'):
    labels_df = pd.read_csv(label_file)
    labels_df.set_index('image_id', inplace=True)

    # Initialize label encoder
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
X, Y, label_encoder = load_data_with_augmentation(
    image_folders=image_folders,
    label_file=label_file,
    target_samples=2000,  # This is the target number of samples per class
    specific_labels=['akiec', 'bcc', 'mel']
)


num_classes = len(np.unique(Y))
num_features = X.shape[1]
W = np.zeros((num_classes, num_features))
alpha = 0.001
nep = 100
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



image_folders = ['./dataverse_files/HAM10000_images_part_1', './dataverse_files/HAM10000_images_part_2']
label_file = './dataverse_files/newHAM100metadata'
X_nocancer, Y_nocancer, label_encoder_nocancer = load_data_with_augmentation(
    image_folders=image_folders,
    label_file=label_file,
    target_samples=2000,  # This is the target number of samples per class
    specific_labels=['bkl', 'df', 'nv',"vasc"]
)



num_classes = len(np.unique(Y_nocancer))
num_features = X_nocancer.shape[1]
W_nocancer = np.zeros((num_classes, num_features))
alpha = 0.001
btch = 256
N = len(Y_nocancer)
nbtch = ceil(N / btch)
beta1, beta2, epsilon = 0.9, 0.999, 1e-8

losses_nocancer = []
m, v = np.zeros_like(W_nocancer), np.zeros_like(W_nocancer)
K = 1
total_correct_predictions = 0

for epoch in range(nep):
    epoch_loss = 0.0
    correct_predictions = 0
    for i in range(0, N, btch):
        end = min(i + btch, N)
        x_batch = X_nocancer[i:end]
        y_batch = Y_nocancer[i:end]
        for j in range(x_batch.shape[0]):
            x_i = x_batch[j][:, np.newaxis]
            y_i = y_batch[j]
            gk = compute_gradient(W_nocancer, x_i, y_i)
            W_nocancer -= alpha * gk  # Simple gradient descent update
            scores = np.dot(W_nocancer, x_i)
            fw = functinonw(scores)
            loss_instance = -np.log(fw[y_i])
            epoch_loss += loss_instance.item()  # Convert numpy array to scalar if needed
            if np.argmax(fw) == y_i:
                correct_predictions += 1
    accuracy = correct_predictions / float(N)
    losses_nocancer.append(epoch_loss / N)  # Ensure losses are collected for plotting
    print(f'Epoch {epoch + 1}/{nep}, Loss: {epoch_loss / N:.4f}, Accuracy: {accuracy:.4f}')

plt.plot(range(1, nep + 1), losses_nocancer)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.show()


image_folders = ['./dataverse_files/HAM10000_images_part_1', './dataverse_files/HAM10000_images_part_2']
label_file = './dataverse_files/newHAM100metadata'
X_cancerous, Y_cancerous, label_encoder_cancerous = load_data_with_augmentation(
    image_folders=image_folders,
    label_file=label_file,
    target_samples=2000,
    column = 'cancerous',
)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # Avoiding division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip values to avoid log(0)
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)

num_classes = len(np.unique(Y_cancerous))
num_features = X_cancerous.shape[1]
W_cancerous = np.zeros((num_classes, num_features))
alpha = 0.001
btch = 256
N = len(Y)
nbtch = ceil(N / btch)
beta1, beta2, epsilon = 0.9, 0.999, 1e-8

losses_cancerous = []
m, v = np.zeros_like(W), np.zeros_like(W)
K = 1
total_correct_predictions = 0

losses_cancerous = []  # List to store losses for each epoch

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # Avoiding division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip values to avoid log(0)
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)

for epoch in range(nep):
    epoch_loss = 0.0
    correct_predictions = 0

    # Iterate over batches
    for i in range(0, N, btch):
        end = min(i + btch, N)
        x_batch = X_cancerous[i:end]
        y_batch = Y_cancerous[i:end]

        # Iterate over batch samples
        for j in range(x_batch.shape[0]):
            x_i = x_batch[j][:, np.newaxis]
            y_i = y_batch[j]
            gk = compute_gradient(W_cancerous, x_i, y_i)
            W_cancerous -= alpha * gk  # Simple gradient descent update

            # Predict and compute loss
            scores = np.dot(W_cancerous, x_i)
            fw = functinonw(scores)
            predicted_probability = fw[1]  # Probability of being cancerous (positive class)
            loss_instance = binary_cross_entropy(y_i, predicted_probability)
            epoch_loss += loss_instance  # No need to convert to scalar, as it's already scalar due to binary cross-entropy

            # Check accuracy
            if np.argmax(fw) == y_i:
                correct_predictions += 1

    # Calculate accuracy and append loss for this epoch
    accuracy = correct_predictions / float(N)
    losses_cancerous.append(epoch_loss / N)

    # Print epoch loss and accuracy
    print(f'Epoch {epoch + 1}/{nep}, Loss: {epoch_loss / N:.4f}, Accuracy: {accuracy:.4f}')


plt.plot(range(1, nep + 1), losses_cancerous)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.show()


from collections import defaultdict

# Load test data
test_images = ['./dataverse_files/HAM10000_images_part_1']
x_test, Y_test, test_label_encoder = load_data(test_images, label_file)
x_test_cancerous, Y_test_cancerous, test_label_encoder_cancerous = load_data(test_images, label_file,
                                                                             column='cancerous')
test_accuracy = 0
correct_test_predictions = 0

# Initialize dictionaries to track predictions
class_correct = defaultdict(int)
class_total = defaultdict(int)

# Process each test image
for i in range(len(x_test)):
    # Predict cancerous attribute
    x_i_cancerous = x_test_cancerous[i].reshape(-1, 1)
    scores_cancerous = np.dot(W_cancerous, x_i_cancerous)
    predicted_cancerous = np.argmax(functinonw(scores_cancerous))

    # Choose appropriate weights based on predicted cancerous value
    if predicted_cancerous == 1:  # Cancerous
        W_to_use = W
        label_encoder_to_use = test_label_encoder
    else:  # Not cancerous
        W_to_use = W_nocancer
        label_encoder_to_use = label_encoder_nocancer

    # Predict label using appropriate weights
    x_i = x_test[i].reshape(-1, 1)
    scores = np.dot(W_to_use, x_i)
    probabilities = functinonw(scores)
    predicted_label_int = np.argmax(probabilities)
    actual_label_int = Y_test[i]

    # Convert numerical labels to original class names for actual and predicted
    actual_label_name = test_label_encoder.inverse_transform([actual_label_int])[0]
    predicted_label_name = label_encoder_to_use.inverse_transform([predicted_label_int])[0]

    # Print each label and its prediction
    print(f'Image {i + 1}:Predict Cancer - {predicted_cancerous} , Actual Cancer {Y_test_cancerous[i]};Actual label - {actual_label_name}, Predicted label - {predicted_label_name}')

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