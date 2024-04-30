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

size = 32

# Function to load and preprocess data
# Function to load and preprocess data
def load_data(image_folders, label_file, img_size=32, specific_labels=None, column = 'dx'):
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




image_folders = ['./dataverse_files/HAM10000_images_part_1', './dataverse_files/HAM10000_images_part_2']
label_file = './dataverse_files/HAM10000_metadata'
X, Y, label_encoder = load_data(
    image_folders=image_folders,
    label_file=label_file,
    specific_labels=['akiec', 'bcc', 'mel']
)


num_classes = len(np.unique(Y))
num_features = X.shape[1]
W = np.zeros((num_classes, num_features))
nep = 1000
btch = 256
N = len(Y)
nbtch = ceil(N / btch)
beta1, beta2, epsilon = 0.9, 0.999, 1e-8
alpha = 0.001


losses = []
accuracies = []
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
            m = beta1 * m + (1 - beta1) * gk
            v = beta2 * v + (1 - beta2) * gk * gk
            m_hat = m / (1 - beta1 ** K)
            v_hat = v / (1 - beta2 ** K)
            W -= alpha * gk  # Simple gradient descent updat
            scores = np.dot(W, x_i)
            fw = functinonw(scores)
            loss_instance = -np.log(fw[y_i])
            epoch_loss += loss_instance.item()  # Convert numpy array to scalar if needed

            if np.argmax(fw) == y_i:
                correct_predictions += 1
    accuracy = correct_predictions / float(N)
    accuracies.append(accuracy)

    losses.append(epoch_loss / N)  # Ensure losses are collected for plotting
    average_loss = epoch_loss / N

    # If average_loss is an array but expected to be scalar, explicitly convert it:
    if isinstance(average_loss, np.ndarray) and average_loss.size == 1:
        average_loss = average_loss.item()  # This converts a one-element array to a scalar

    total_correct_predictions += correct_predictions
    print(f'Epoch {epoch + 1}/{nep}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')


#plt.plot(range(1, nep + 1), losses)
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.title('Loss per Epoch Two Step Adam Dynamic Learning Rate')
#plt.show()



image_folders = ['./dataverse_files/HAM10000_images_part_1', './dataverse_files/HAM10000_images_part_2']
label_file = './dataverse_files/newHAM100metadata'
X_nocancer, Y_nocancer, label_encoder_nocancer = load_data(
    image_folders=image_folders,
    label_file=label_file,
    specific_labels=['bkl', 'df', 'nv',"vasc"]
)


num_classes = len(np.unique(Y_nocancer))
num_features = X_nocancer.shape[1]
W_nocancer = np.zeros((num_classes, num_features))
btch = 256
N = len(Y_nocancer)
nbtch = ceil(N / btch)
beta1, beta2, epsilon = 0.9, 0.999, 1e-8

losses_nocancer = []
accuracies_nocancer = []
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
            m = beta1 * m + (1 - beta1) * gk
            v = beta2 * v + (1 - beta2) * gk * gk
            m_hat = m / (1 - beta1 ** K)
            v_hat = v / (1 - beta2 ** K)
            W_nocancer -= alpha * gk  # Simple gradient descent updat
            scores = np.dot(W_nocancer, x_i)
            fw = functinonw(scores)
            loss_instance = -np.log(fw[y_i])
            epoch_loss += loss_instance.item()  # Convert numpy array to scalar if needed

            if np.argmax(fw) == y_i:
                correct_predictions += 1
    accuracy_nocancer = correct_predictions / float(N)
    accuracies_nocancer.append(accuracy_nocancer)
    losses_nocancer.append(epoch_loss / N)  # Ensure losses are collected for plotting
    average_loss = epoch_loss / N

    # If average_loss is an array but expected to be scalar, explicitly convert it:
    if isinstance(average_loss, np.ndarray) and average_loss.size == 1:
        average_loss = average_loss.item()  # This converts a one-element array to a scalar

    total_correct_predictions += correct_predictions
    print(f'Epoch {epoch + 1}/{nep}, Loss: {average_loss:.4f}, Accuracy: {accuracy_nocancer:.4f}')

#plt.plot(range(1, nep + 1), losses_nocancer)
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.title('Loss per Epoch')
#plt.show()


image_folders = ['./dataverse_files/HAM10000_images_part_1', './dataverse_files/HAM10000_images_part_2']
label_file = './dataverse_files/newHAM100metadata'
X_cancerous, Y_cancerous, label_encoder_cancerous = load_data(
    image_folders=image_folders,
    label_file=label_file,
    column = 'cancerous',
)


num_classes = len(np.unique(Y_cancerous))
num_features = X_cancerous.shape[1]
btch = 256
N = len(Y)
nbtch = ceil(N / btch)
beta1, beta2, epsilon = 0.9, 0.999, 1e-8

losses_cancerous = []
accuracies_cancerous = []
m, v = np.zeros_like(W), np.zeros_like(W)
K = 1
total_correct_predictions = 0

losses_cancerous = []  # List to store losses for each epoch
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_gradient_binary(W, x_i, y_i):
    """
    Compute gradient for binary classification.
    W: weight matrix (num_features, 1)
    x_i: input features (num_features, 1)
    y_i: actual label (scalar, 0 or 1)
    """
    scores = np.dot(W.T, x_i)  # Transpose W to match dimensions
    p = sigmoid(scores)  # Compute the sigmoid probability
    error = p - y_i  # Difference between predicted and actual
    grad = x_i * error  # Gradient computation
    return grad

threshold = 0.45

# Assuming W_cancerous should be (number of features, 1) for binary classification
W_cancerous = np.zeros((num_features, 1))
m = np.zeros((num_features, 1))
v = np.zeros((num_features, 1))

for epoch in range(nep):
    epoch_loss = 0.0
    correct_predictions = 0
    for i in range(0, N, btch):
        end = min(i + btch, N)
        x_batch = X_cancerous[i:end]
        y_batch = Y_cancerous[i:end]
        for j in range(x_batch.shape[0]):
            x_i = x_batch[j, :].reshape(-1, 1)  # Ensure x_i is a column vector
            y_i = y_batch[j]
            gk = compute_gradient_binary(W_cancerous, x_i, y_i)
            m = beta1 * m + (1 - beta1) * gk
            v = beta2 * v + (1 - beta2) * gk * gk
            m_hat = m / (1 - beta1 ** K)
            v_hat = v / (1 - beta2 ** K)
            W_cancerous -= alpha * gk  # Update weights
            scores = np.dot(W_cancerous.T, x_i)
            probability = sigmoid(scores)
            loss_instance = -y_i * np.log(probability) - (1 - y_i) * np.log(1 - probability)
            epoch_loss += loss_instance

            if (probability > threshold).astype(int) == y_i:
                correct_predictions += 1

    # After processing all batches, calculate and print epoch summary
    accuracy_cancerous = correct_predictions / float(N)
    accuracies_cancerous.append(accuracy_cancerous)
    average_loss = epoch_loss / N
    losses_cancerous.append(average_loss)

    if isinstance(average_loss, np.ndarray) and average_loss.size == 1:
        average_loss = average_loss.item()

    print(f'Epoch {epoch + 1}/{nep}, Loss: {average_loss:.4f}, Accuracy: {accuracy_cancerous:.4f}')



#plt.plot(range(1, nep + 1), losses_cancerous)
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.title('Loss per Epoch')
#plt.show()

from collections import defaultdict

# Load test data
test_images = ['./dataverse_files/ISIC2018_Task3_Test_Images']
label_file = './dataverse_files/newTest'
x_test, Y_test, test_label_encoder = load_data(test_images, label_file)
x_test_cancerous, Y_test_cancerous, test_label_encoder_cancerous = load_data(test_images, label_file, column='cancerous')
test_accuracy = 0
correct_test_predictions = 0

# Initialize dictionaries to track predictions for detailed accuracy reports
class_correct = defaultdict(int)
class_total = defaultdict(int)
correct_cancerous_predictions = 0
total_cancerous_predictions = 0

# Process each test image
for i in range(len(x_test)):
    # Predict if image is cancerous
    x_i_cancerous = x_test_cancerous[i].reshape(-1, 1)
    scores_cancerous = np.dot(W_cancerous.T, x_i_cancerous)  # Transpose W_cancerous
    probability_cancerous = sigmoid(scores_cancerous)

    # Use threshold to predict class
    predicted_cancerous = (probability_cancerous >= threshold).astype(int)
    total_cancerous_predictions += 1
    if predicted_cancerous == Y_test_cancerous[i]:
        correct_cancerous_predictions += 1

    # Choose appropriate weights based on predicted cancerous value
    W_to_use = W if predicted_cancerous == 1 else W_nocancer
    label_encoder_to_use = label_encoder if predicted_cancerous == 1 else label_encoder_nocancer

    # Predict label using the selected weights
    x_i = x_test[i].reshape(-1, 1)
    scores = np.dot(W_to_use, x_i)
    probabilities = functinonw(scores)
    predicted_label_int = np.argmax(probabilities)
    actual_label_int = Y_test[i]

    # Convert numerical labels to original class names for actual and predicted
    actual_label_name = test_label_encoder.inverse_transform([actual_label_int])[0]
    predicted_label_name = label_encoder_to_use.inverse_transform([predicted_label_int])[0]

    # Print each label and its prediction
    print(f'Image {i + 1}: Predicted Cancer - {predicted_cancerous}, Actual Cancer - {Y_test_cancerous[i]}; Actual label - {actual_label_name}, Predicted label - {predicted_label_name}')

    # Update prediction tracking
    class_total[actual_label_name] += 1
    if predicted_label_name == actual_label_name:
        correct_test_predictions += 1
        class_correct[actual_label_name] += 1

# Calculate overall test accuracy
test_accuracy = correct_test_predictions / len(x_test) if len(x_test) > 0 else 0
binary_accuracy = correct_cancerous_predictions / total_cancerous_predictions if total_cancerous_predictions > 0 else 0
print(f"Overall Test Accuracy: {test_accuracy:.4f}")
print(f"Binary Test Accuracy (Cancerous Prediction): {binary_accuracy:.4f}")

# Calculate and print accuracy for each class
for label_name, total in class_total.items():
    accuracy = class_correct[label_name] / total if total > 0 else 0
    print(f'Accuracy for {label_name}: {accuracy:.4f}')




epochs = list(range(1, nep + 1))

print(losses)
print(losses_nocancer)

# Assuming losses_cancerous is a list of NumPy arrays
losses_cancerous = [item for array in losses_cancerous for item in array.flatten()]
print(losses_cancerous)


plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, label='Cancerous Classes')
plt.plot(epochs, losses_nocancer, label='Non-cancerous classes')
plt.plot(epochs, losses_cancerous, label='Binary Classification')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch for Different Models')
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracies, label='Cancerous Classes')
plt.plot(epochs, accuracies_nocancer, label='Non-cancerous classes')
plt.plot(epochs, accuracies_cancerous, label='Binary Classification')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Accuracy per Epoch for Different Models')
plt.legend()
plt.grid(True)
plt.show()