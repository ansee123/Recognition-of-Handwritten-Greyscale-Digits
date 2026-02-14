#importing libraries
import zipfile
import pandas as pd
import os
from PIL import Image
import struct
from array import array
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import cv2
import argparse
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import pickle
import time
from matplotlib import style
# Path to your ZIP file
zip_path = "/content/dataset.zip"
extract_path = "/content/extracted_dataset"

# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Collect all files
all_files = []
for root, dirs, files in os.walk(extract_path):
    for file in files:
        all_files.append(os.path.join(root, file))

print(f" Extracted {len(all_files)} files from ZIP.\n")

# Detect file types
img_files = [f for f in all_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
mnist_files = [f for f in all_files if "ubyte" in f.lower()]
if mnist_files:
    print("MNIST binary format detected.")

    def load_mnist_images(filename):
        with open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            data = array("B", f.read())
        return num, rows, cols

    img_file = [f for f in mnist_files if "images" in f][0]
    num, rows, cols = load_mnist_images(img_file)
    print(f"Number of images: {num}")
    print(f"Image size: {rows}x{cols}")
#checking null values present in the dataset
# Path to extracted dataset
path = "/content/extracted_dataset"

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows, cols)
        return data

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Load training set
train_images = load_mnist_images(os.path.join(path, "train-images-idx3-ubyte"))
train_labels = load_mnist_labels(os.path.join(path, "train-labels-idx1-ubyte"))

# Load test set
test_images = load_mnist_images(os.path.join(path, "t10k-images-idx3-ubyte"))
test_labels = load_mnist_labels(os.path.join(path, "t10k-labels-idx1-ubyte"))

# Check for null/NaN values
print("Train images nulls:", np.isnan(train_images).sum())
print("Train labels nulls:", np.isnan(train_labels).sum())
print("Test images nulls:", np.isnan(test_images).sum())
print("Test labels nulls:", np.isnan(test_labels).sum())# checking outliers present
# Flatten images to 1D arrays (each row = one image's pixels)
train_pixels = train_images.reshape(train_images.shape[0], -1)
test_pixels = test_images.reshape(test_images.shape[0], -1)

# Take a random sample (for speed, otherwise 60k images may be too heavy)
sample_train = train_pixels[np.random.choice(train_pixels.shape[0], 2000, replace=False)]
sample_test = test_pixels[np.random.choice(test_pixels.shape[0], 1000, replace=False)]

# Create boxplot
plt.figure(figsize=(8,6))
plt.boxplot([sample_train.flatten(), sample_test.flatten()],
            labels=["Train Pixels", "Test Pixels"],
            showfliers=True)  # showfliers=True marks outliers
plt.title("Boxplot of Pixel Intensity Distribution (MNIST)")
plt.ylabel("Pixel Value (0–255)")
plt.show()
# Load MNIST images and labels
(images, labels), (_, _) = mnist.load_data()

# Show first 5 images
for i in range(5):
    plt.imshow(images[i], cmap="gray")
    plt.title(f"Label: {labels[i]}")
    plt.axis("off")
    plt.show()# Assuming you already have train_labels and test_labels loaded as numpy arrays
unique_train, counts_train = np.unique(train_labels, return_counts=True)
unique_test, counts_test = np.unique(test_labels, return_counts=True)

# Plot per-class distribution
plt.figure(figsize=(10,6))
plt.bar(unique_train, counts_train, alpha=0.7, label="Train")
plt.bar(unique_test, counts_test, alpha=0.7, label="Test")
plt.title("Digit Class Distribution in MNIST Dataset")
plt.xlabel("Digit Class (0-9)")
plt.ylabel("Number of Samples")
plt.legend()
plt.show()
# CNN Model

# ------------------- Build CNN model -------------------
def build_cnn(width, height, depth, total_classes):
    model = Sequential()
    # Conv layer 1
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(height, width, depth)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Conv layer 2
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten & Dense layers
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(total_classes, activation="softmax"))
    return model

# ------------------- Argument parser -------------------
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save_model", type=int, default=-1)
ap.add_argument("-l", "--load_model", type=int, default=-1)
ap.add_argument("-w", "--save_weights", type=str, default="mnist_cnn_weights.h5")
args, unknown = ap.parse_known_args()

# ------------------- Load dataset -------------------
print('Loading MNIST Dataset...')
# dataset = fetch_openml('mnist_784', version=1, as_frame=False) # Removed this line
(train_img, train_labels), (test_img, test_labels) = mnist.load_data() # Added this line

train_img = train_img.reshape((train_img.shape[0], 28, 28, 1)) # Added this line for shape compatibility
test_img = test_img.reshape((test_img.shape[0], 28, 28, 1)) # Added this line for shape compatibility

train_img = train_img.astype('float32') / 255.0 # Added this line for normalization
test_img = test_img.astype('float32') / 255.0 # Added this line for normalization


train_labels = to_categorical(train_labels, 10) # Changed np_utils.to_categorical to to_categorical
test_labels = to_categorical(test_labels, 10) # Changed np_utils.to_categorical to to_categorical

# ------------------- Compile model -------------------
print('\nCompiling model...')
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
clf = build_cnn(width=28, height=28, depth=1, total_classes=10)
clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# ------------------- Train or load weights -------------------
if args.load_model < 0:
    print('\nTraining the Model...')
    clf.fit(train_img, train_labels, batch_size=128, epochs=5, verbose=1)
    loss, accuracy = clf.evaluate(test_img, test_labels, batch_size=128, verbose=1)
    print(f'Accuracy of Model: {accuracy :}')

if args.save_model > 0:
    print('Saving weights to file...')
    clf.save_weights(args.save_weights)
#KNN Model

style.use('ggplot')

print('\nLoading MNIST Data...')
(train_img, train_labels), (test_img, test_labels) = mnist.load_data()

# Flatten images for KNN
train_img = train_img.reshape((train_img.shape[0], -1))
test_img = test_img.reshape((test_img.shape[0], -1))

# Prepare train/validation split
X_train, X_test, y_train, y_test = train_test_split(train_img, train_labels, test_size=0.1)

clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto', n_jobs=10)
clf.fit(X_train, y_train)

# Save model
with open('MNIST_KNN.pickle', 'wb') as f:
    pickle.dump(clf, f)

y_pred = clf.predict(X_test)
print('Accuracy of Model:', accuracy_score(y_test, y_pred))# SVM

print("\nLoading MNIST Data...")
(train_img, train_labels), (test_img, test_labels) = mnist.load_data()

# Flatten images from 28x28 to 784
train_img = train_img.reshape((train_img.shape[0], -1))
test_img = test_img.reshape((test_img.shape[0], -1))

# Normalize pixel values to 0-1
train_img = train_img / 255.0
test_img = test_img / 255.0

# Prepare classifier training/validation data
X_train, X_val, y_train, y_val = train_test_split(
    train_img, train_labels, test_size=0.1
)
clf = svm.SVC(gamma=0.1, kernel="poly")
clf.fit(X_train, y_train)

# Save the trained model
with open("MNIST_SVM.pickle", "wb") as f:
    pickle.dump(clf, f)

# Load model back
with open("MNIST_SVM.pickle", "rb") as f:
    clf = pickle.load(f)


# ===== Test Accuracy =====
y_pred_test = clf.predict(test_img)
test_acc = accuracy_score(test_labels, y_pred_test)

print("\nAccuracy of Model:", test_acc)#RANDOM FOREST

# Open log file
log_file = open("summary.log", "w")

def log_and_print(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)

log_and_print('\nLoading MNIST Data...')
(train_img, train_labels), (test_img, test_labels) = mnist.load_data()

# Flatten & normalize
train_img = train_img.reshape(train_img.shape[0], -1) / 255.0
test_img = test_img.reshape(test_img.shape[0], -1) / 255.0

# Split train/validation
X_train, X_val, y_train, y_val = train_test_split(
    train_img, train_labels, test_size=0.1
)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf.fit(X_train, y_train)

# Save model
with open('MNIST_RFC.pickle', 'wb') as f:
    pickle.dump(clf, f)

# Load model back
with open('MNIST_RFC.pickle', 'rb') as f:
    clf = pickle.load(f)
# Test results
y_pred_test = clf.predict(test_img)
test_acc = accuracy_score(test_labels, y_pred_test)
test_conf_mat = confusion_matrix(test_labels, y_pred_test)
log_and_print('\nAccuracy of Model:', test_acc)

# Close log file
log_file.close()
# ---------------- Load MNIST ----------------
print("Loading MNIST...")
(train_img, train_labels), (test_img, test_labels) = mnist.load_data()

# Normalize
train_img = train_img / 255.0
test_img = test_img / 255.0

# Flatten for ML models
X_train_flat = train_img.reshape(train_img.shape[0], -1)
X_test_flat = test_img.reshape(test_img.shape[0], -1)

# ---------------- Split for validation ----------------
X_train, X_val, y_train, y_val = train_test_split(X_train_flat, train_labels, test_size=0.1, random_state=42)

results = []

# ---------------- KNN ----------------
start = time.time()
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
acc_knn = accuracy_score(test_labels, knn.predict(X_test_flat))
results.append(("KNN", acc_knn, time.time() - start))

# ---------------- SVM ----------------
start = time.time()
svm_clf = svm.SVC(gamma=0.1, kernel="poly")
svm_clf.fit(X_train, y_train)
acc_svm = accuracy_score(test_labels, svm_clf.predict(X_test_flat))
results.append(("SVM (poly)", acc_svm, time.time() - start))

# ---------------- Random Forest ----------------
start = time.time()
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)
acc_rf = accuracy_score(test_labels, rf.predict(X_test_flat))
results.append(("Random Forest", acc_rf, time.time() - start))

# ---------------- CNN ----------------
train_img_cnn = np.expand_dims(train_img, -1)
test_img_cnn = np.expand_dims(test_img, -1)
y_train_cnn = to_categorical(train_labels, 10)
y_test_cnn = to_categorical(test_labels, 10)

cnn = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
start = time.time()
cnn.fit(train_img_cnn, y_train_cnn, epochs=3, batch_size=128, verbose=1)
_, acc_cnn = cnn.evaluate(test_img_cnn, y_test_cnn, verbose=0)
results.append(("CNN", acc_cnn, time.time() - start))

# ---------------- Print Results ----------------
print("\nModel Comparison:")
print("{:<15} {:<15} {:<15}".format("Model", "Accuracy", "Train Time (s)"))
print("-" * 45)
for model, acc, t in results:
    print("{:<15} {:<15.4f} {:<15.2f}".format(model, acc, t))
# Save the CNN  model in HDF5 format
cnn.save("cnn_mnist_model.h5")