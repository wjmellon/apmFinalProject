import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import os
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

def compute_median_frequency_weights(labels):
    """
    Computes weights for each class based on Median Frequency Balancing.
    :param labels: array of integer class labels
    :return: dictionary of class weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    total_count = sum(counts)
    frequency = counts / total_count
    median_freq = np.median(frequency)
    weights = median_freq / frequency
    return {k: v for k, v in zip(unique, weights)}

# Calculate the class weights using Median Frequency Balancing
def compute_median_frequency_weights(labels):
    """
    Computes weights for each class based on Median Frequency Balancing.
    :param labels: array of integer class labels
    :return: dictionary of class weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    total_count = sum(counts)
    frequency = counts / total_count
    median_freq = np.median(frequency)
    weights = median_freq / frequency
    return {k: v for k, v in zip(unique, weights)}



def focal_loss(probabilities, true_class, alpha=0.25, gamma=2.0, epsilon=1e-8):
    p_t = probabilities[true_class] + epsilon  # Adding epsilon to avoid log(0)
    alpha_t = alpha if true_class == 1 else (1 - alpha)

    # Focal loss calculation with clipped probabilities
    loss = -alpha_t * ((1 - p_t) ** gamma) * np.log(p_t)
    return loss

# Function to compute softmax probabilities
def functinonw(z):
    ez = np.exp(z - np.max(z, axis=0, keepdims=True))
    return ez / np.sum(ez, axis=0, keepdims=True)


# Function to compute gradient for updating weights
def compute_gradient(W, x_i, y_i):
    scores = np.dot(W, x_i)
    probabilities = functinonw(scores)
    probabilities[y_i] -= 1  # Correct for the one-hot encoded target
    grad = np.outer(probabilities, x_i)
    return grad

def load_data(image_folders, label_file, img_size=112, undersample_count=None, is_training=True):
    labels_df = pd.read_csv(label_file)
    labels_df.set_index('image_id', inplace=True)

    # Initialize label encoder
    label_encoder = LabelEncoder()
    all_labels = labels_df['dx'].unique()
    label_encoder.fit(all_labels)

    images = []
    label_list = []
    missing_labels = []

    if is_training:
        # Calculate counts of each label
        label_counts = labels_df['dx'].value_counts()

        # If undersample_count is not set, use the smallest non-'nv' class count for undersampling
        if undersample_count is None:
            undersample_count = label_counts[label_counts.index != 'nv'].min()

        # Generate a random sample of indices from the 'nv' category
        nv_indices = labels_df[labels_df['dx'] == 'nv'].sample(n=undersample_count, random_state=42).index
    else:
        nv_indices = labels_df.index  # Use all indices if not training

    # Process each image folder
    for image_folder in image_folders:
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

        for file in image_files:
            image_id = file.split('.')[0]
            if image_id in labels_df.index:
                label = labels_df.loc[image_id, 'dx']
                if is_training and label == 'nv' and image_id not in nv_indices:
                    continue  # Skip 'nv' images not in the sampled indices if training
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

# Usage of the function with undersampling
# Assuming 'mel' has 1113 instances and you want to undersample 'nv' to this count

# Main script to execute training
image_folders = [
    './dataverse_files/HAM10000_images_part_1',
    './dataverse_files/HAM10000_images_part_2'
]
label_file = './dataverse_files/HAM10000_metadata'
X, Y, label_encoder = load_data(image_folders, label_file,112, 1113)
class_weights_mfb = compute_median_frequency_weights(Y)

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

# Compute class weights
#unique_classes = np.unique(Y)
# First, get the base class weights using sklearn's compute_class_weight for balanced classes
#base_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
# class_weights = {k: v for k, v in zip(np.unique(Y), base_weights)}

# Now manually increase the weight for the 'mel' class.
# The multiplier for the 'mel' class's weight can be tuned according to how much emphasis you want to put on that class.
# This is a hyperparameter that you would typically tune based on your validation set performance.

# mel_class_index = label_encoder.transform(['mel'])[0] # Get the class index for 'mel'
# nv_class_index = label_encoder.transform(['nv'])[0]   # Get the class index for 'nv'
# vasc_class_index = label_encoder.transform(['vasc'])[0]
# You could, for example, double the weight for the 'mel' class
# class_weights[mel_class_index] *= 100

# Now reduce the weight for the 'nv' class if you think it's still being overemphasized
# This should be done cautiously to avoid underemphasizing 'nv' to the point where the model fails to learn it
#class_weights[nv_class_index] *= 0.2
#class_weights[vasc_class_index] *= 0.1



losses = []
m, v = np.zeros_like(W), np.zeros_like(W)
K = 1
# Adjust the gradient update step in the training loop
for epoch in range(nep):
    epoch_loss = 0
    for i in range(int(nbtch)):
        batch_start = i * btch
        batch_end = min(batch_start + btch, N)
        for j in range(batch_start, batch_end):
            x_i = X[j].reshape(-1, 1)
            y_i = Y[j]
            gk = compute_gradient(W, x_i, y_i)

            # Apply the class weight
            class_weight = class_weights_mfb[y_i]
            gk = gk * class_weight  # Scale the gradient by the corresponding class weight

            m = beta1 * m + (1 - beta1) * gk
            v = beta2 * v + (1 - beta2) * (gk ** 2)
            m_hat = m / (1 - beta1 ** K)
            v_hat = v / (1 - beta2 ** K)
            W -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

            scores = np.dot(W, x_i)
            probabilities = functinonw(scores)
            loss = focal_loss(probabilities, y_i)  # Compute focal loss without class weight scaling

            # Scale the loss by the class weight after computation
            weighted_loss = loss * class_weight
            epoch_loss += weighted_loss.item()  # Add weighted loss to epoch loss
            K += 1

    epoch_loss /= N  # Average the epoch loss
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
x_test, Y_test, test_label_encoder = load_data(test_images, label_file,img_size=112, is_training=False)
# Predict on the first 10 images of the validation set
# Predict on the available images in the validation set
num_images_available = len(x_test)  # Get the actual number of images loaded
num_images_to_predict = min(num_images_available,25)  # Predict on up to 10 images, or however many are available

for i in range(num_images_to_predict):
    x_i = x_test[i].reshape(-1, 1)
    scores = np.dot(W, x_i)
    probabilities = functinonw(scores)
    predicted_label_int = np.argmax(probabilities)
    actual_label_int = Y_test[i]

    # Use inverse_transform to convert the integer labels back to original string labels
    predicted_label_str = label_encoder.inverse_transform([predicted_label_int])[0]
    actual_label_str = label_encoder.inverse_transform([actual_label_int])[0]

    print(f'Image {i + 1}: Predicted Label: {predicted_label_str}, Actual Label: {actual_label_str}')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch with α = ' + str(alpha) + ' Batch Size = '+str(btch) +' Adam')

plt.show()


image_index = 8  # Example index, adjust as needed
# Assuming x_test contains flat arrays of RGB images with each image of size 112x112x3
# and each image array being of length 112*112*3 = 37632
image_data = x_test[image_index].reshape(112, 112, 3)  # Reshape the data to 112x112 pixels with 3 channels

plt.imshow(image_data)  # Plot the image in color
plt.title(f'Image Label: {label_encoder.inverse_transform([Y_test[image_index]])[0]}')  # Optional: Set title with the label of the image
plt.colorbar()  # Optional: To show the color bar, though it might not be informative for RGB images
plt.show()
