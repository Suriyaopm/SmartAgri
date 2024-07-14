	import os
	import numpy as np
	from sklearn.model_selection import train_test_split
	from sklearn.svm import SVC
	from sklearn.metrics import accuracy_score
	from sklearn.preprocessing import StandardScaler
	from sklearn.pipeline import Pipeline
	from skimage.io import imread
	from skimage.transform import resize
	
	# Replace these paths with the paths to your dataset
	dataset_path = 'path_to_dataset_folder'
	disease_classes = os.listdir(dataset_path)

	# Load and preprocess the dataset
	X = []
	y = []

	for class_idx, disease_class in enumerate(disease_classes):
   	    class_path = os.path.join(dataset_path, disease_class)
    	for image_name in os.listdir(class_path):
       	    image_path = os.path.join(class_path, image_name)
        	image = imread(image_path)
       	    resized_image = resize(image, (100, 100))  # Resize the image to a consistent size
       	    X.append(resized_image.flatten())
       	    y.append(class_idx)

	X = np.array(X)
	y = np.array(y)

	# Split the dataset into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Create a pipeline with preprocessing and SVM classifier
	svm_model = Pipeline([
   	    ('scaler', StandardScaler()),
    	    ('svm', SVC(kernel='linear', C=1.0))
	])

	# Train the SVM model
	svm_model.fit(X_train, y_train)

	# Predict disease classes on the test set
	y_pred = svm_model.predict(X_test)

	# Calculate accuracy
	accuracy = accuracy_score(y_test, y_pred)
	print(f"Accuracy: {accuracy:.2f}")
