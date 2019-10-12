# k-Nearest Neighbors (kNN)

Implementation of k-Nearest Neighbors algorithm for classifying MNIST dataset.
This is implemented simply by using numpy (no libraries) in Python 3.6.7

# Random Forest (using sklearn library)

An ensemble learning classification model. Constructs a multitude of decision trees at training time and outputs the class at the mode of the classes.
Random forest is essentially a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to
improve the predict accuracy and control overfitting. The sub-sample size M is always the same as the original input size N (M=N).
The sample are bootstrapped with replacement.

## References

For further studies about kNN, refer to the following websites.

    https://www.geeksforgeeks.org/implementation-k-nearest-neighbors/
    https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

For further studies about random forest algorithm, refer to the following websites.

    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.fit
    https://towardsdatascience.com/understanding-random-forest-58381e0602d2