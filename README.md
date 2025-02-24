# **Cat vs. Dog Classification Using SVM and OpenCV** #

### 1. Importing Necessary Libraries ###
The script imports **NumPy, OpenCV, Matplotlib, Scikit-learn,** and **TQDM** for:
- Handling image processing (`cv2`)
- Managing arrays (`numpy`)
- Splitting data & training a machine learning model (`sklearn`)
- Displaying progress bars (`tqdm`)
  
### 2. Loading and Preprocessing the Dataset ###
- The dataset is stored in `"C:/Users/KIIT/Desktop/PetImages"`, with subfolders `"Cat"` and `"Dog"`.
- The `load_data()` function:
  - Reads images in grayscale (`cv2.IMREAD_GRAYSCALE`).
  - Resizes each image to **64x64** pixels.
  - Assigns labels (`0` for **Cat**, `1` for **Dog**).
  - Skips unreadable/corrupt images.
  - Stores images as **NumPy arrays**.
  
### 3. Removing Corrupt Images ###
- A separate loop scans the dataset folders and removes unreadable files.

### 4. Data Normalization & Reshaping ###
- The pixel values are **normalized** to `[0,1]` by dividing by `255.0`.
- The images are **flattened** into 1D arrays (`64x64 → 4096 features`) to match SVM's input format.
  
### 5. Splitting the Data ###
- The dataset is split into **training (80%)** and **testing (20%)** sets.
  
### 6. Training the SVM Model ###
- **Support Vector Machine (SVM)** with a **linear kernel** is trained using only 500 samples (`X_train_small` and `y_train_small`).
  
### 7. Model Evaluation ###
- The trained model predicts labels for the test set.
- **Accuracy** is calculated using `accuracy_score()`.
  
### 8. Predicting a Single Image ###
- The `predict_image()` function:
  - Loads an image.
  - Resizes and normalizes it.
  - Flattens the image into a vector.
  - Uses the trained **SVM model** to classify it as `"Cat"` or `"Dog"`.
    
### 9. Running a Sample Prediction ###
- The script tests the model with `"C:/Users/KIIT/Desktop/PetImages/Cat/0.jpg".`

## Key Objective:
The key objective of this project is to **classify images of cats and dogs using a Support Vector Machine (SVM) model** with OpenCV for image processing. The goal is to preprocess image data, train an SVM classifier on extracted features, and evaluate its performance in distinguishing between cats and dogs.
