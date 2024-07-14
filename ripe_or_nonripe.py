import numpy as np 
	from sklearn.model_selection import train_test_split 
	from sklearn.tree import DecisionTreeClassifier 
	from sklearn.metrics import accuracy_score 
	from sklearn.preprocessing import LabelEncoder 
	X = np.array([[0, 5, 0],
		          [1, 3, 1],
		          [1, 4, 0],
		          [0, 2, 1],
		          [1, 6, 0]])
	# Labels: 0=unripe, 1=ripe 
	y = np.array([0, 1, 1, 0, 1]) 
	# Split the data into training and testing sets 
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	 # Create and train a Decision Tree classifier 
	clf = DecisionTreeClassifier() 
	clf.fit(X_train, y_train) 
	# Make predictions on the test set 
	y_pred = clf.predict(X_test) 
	# Calculate accuracy 
	accuracy = accuracy_score(y_test, y_pred) 
	print(f"Accuracy: {accuracy:.2f}") 
	# Classify a new tomato (replace with your own tomato's features) 
	new_tomato = np.array([[0, 4, 0]]) 
	prediction = clf.predict(new_tomato) 
	if prediction[0] == 0:
		 print("The tomato is unripe.") 
	else: 
		print("The tomato is ripe.")
