# k-Nearest-Neighbors

	Classification Dataset MNIST
		One of the most well-known datasets in computer vision, consisting of
		images of handwritten digits from 0 to 9. We will be working with a subset of the official version of
		MNIST, denoted as mnist_subset. In particular, we randomly sampled 700 images from each category
		and split them into training, validation, and test sets. This subset corresponds to a JSON file named
		mnist_subset.json. JSON is a lightweight data-interchange format, similar to a dictionary. After loading
		the file, you can access its training, validation, and test splits using the keys ‘train’, ‘valid’, and ‘test’,
		respectively. For example, if we load mnist_subset.json to the variable x, x['train'] refers to the training set
		of mnist_subset. This set is a list with two elements: x['train'][0] containing the features of size
		N (samples) * D (dimension of features), and x['train'][1] containing the corresponding labels of size N.

	Problem 2.1 Distance calculation
		Compute the distance between test data points in X and training data points in Xtrain based on d(xi, xj) = ||xi - xj||2
			function compute_distances(Xtrain, X).

	Problem 2.2 kNN classifier
		Implement kNN classifier. Your algorithm should output the predictions for the test set. 
		Important: You do not need to worry about ties in distance when finding the k nearest neighbor set.
		However, when there are ties in the majority label of the k nearest neighbor set, you should return the label
		with the smallest index. For example, when k = 5, if the labels of the 5 nearest neighbors happen to be
		1, 1, 2, 2, 7, your prediction should be the digit 1.
			function predict_labels(k, ytrain, dists).

	Problem 2.3 Report the accuracy
		The classification accuracy is defined as:
		accuracy = # of correctly classified test examples / # of test examples
		The accuracy value should be in the range of [0,1]
			function compute_accuracy(y, ypred).

	Problem 2.4 Tuning k (6 points)
		Find k among [1, 3, 5, 7, 9] that gives the best classification accuracy on the validation set.
			function find_best k(K, ytrain, dists, yval).