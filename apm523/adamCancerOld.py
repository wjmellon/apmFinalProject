import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import os
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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

size = 112
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


# Main script to execute training
image_folders = [
    './dataverse_files/HAM10000_images_part_1',
    './dataverse_files/HAM10000_images_part_2'
]
label_file = './dataverse_files/HAM10000_metadata'
X, Y, label_encoder = load_data(image_folders, label_file)


# Training setup
num_classes = len(np.unique(Y))
num_features = X.shape[1]
W = np.zeros((num_classes, num_features))
alpha = 0.001  # Learning rate
nep = 20  # Number of epochs
btch = 256  # Batch size
N = len(Y)
nbtch = ceil(N / btch)
beta1, beta2, epsilon = 0.9, 0.999, 1e-8  # Adam parameters

# Training loop with Adam optimizer
losses = []
m, v = np.zeros_like(W), np.zeros_like(W)
K = 1
for epoch in range(nep):
    epoch_loss = 0
    for i in range(int(nbtch)):
        batch_start = i * btch
        batch_end = min(batch_start + btch, N)
        for j in range(batch_start, batch_end):
            x_i = X[j].reshape(-1, 1)
            y_i = Y[j]
            gk = compute_gradient(W, x_i, y_i)
            m = beta1 * m + (1 - beta1) * gk
            v = beta2 * v + (1 - beta2) * (gk * gk)
            m_hat = m / (1 - beta1 ** K)
            v_hat = v / (1 - beta2 ** K)
            W -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)
            scores = np.dot(W, x_i)
            fw = functinonw(scores)
            loss = -np.log(fw[y_i])
            epoch_loss += loss
            K += 1
    epoch_loss /= N
    losses.append(epoch_loss)
    print(f'Epoch {epoch + 1}/{nep}, Loss: {epoch_loss}')

# Output training process
plt.plot(range(1, nep + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.show()

test_images = [
    './tests'
]
x_test, Y_test, test_label_encoder = load_data(test_images, label_file)
# Predict on the first 10 images of the validation set
# Predict on the first 10 images of the validation set
for i in range(10):
    x_i = x_test[i].reshape(-1, 1)
    scores = np.dot(W, x_i)
    probabilities = functinonw(scores)
    predicted_label_int = np.argmax(probabilities)
    actual_label_int = Y_test[i]

    # Use inverse_transform to convert the integer labels back to original string labels
    predicted_label_str = label_encoder.inverse_transform([predicted_label_int])[0]
    actual_label_str = label_encoder.inverse_transform([actual_label_int])[0]

    print(f'Image {i + 1}: Predicted Label: {predicted_label_str}, Actual Label: {actual_label_str}')

plt.plot(range(1, nep + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch with α = ' + str(alpha) + ' Batch Size = '+str(btch) +' Adam')

plt.show()


image_index = 8  # Example index, adjust as needed
# Assuming x_test contains flat arrays of RGB images with each image of size 112x112x3
# and each image array being of length 112*112*3 = 37632
image_data = x_test[image_index].reshape(size, size, 3)  # Reshape the data to 112x112 pixels with 3 channels

plt.imshow(image_data)  # Plot the image in color
plt.title(f'Image Label: {label_encoder.inverse_transform([Y_test[image_index]])[0]}')  # Optional: Set title with the label of the image
plt.colorbar()  # Optional: To show the color bar, though it might not be informative for RGB images
plt.show()
