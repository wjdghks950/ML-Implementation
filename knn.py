from mnist_data_python3 import load_data
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='k-Nearest Neighbors params')
parser.add_argument('-max_test', default=1000, type=int)
parser.add_argument('-max_train', default=50000, type=int)
parser.add_argument('-eigdim', default=2, type=int) # n-dim eigenspace
parser.add_argument('-k', default=5, type=int)
parser.add_argument('-project', default=True, type=bool) #project on eigenspace
args = parser.parse_args()

class KNN():
    def __init__(self, k, train_set, test_set):
        self.k = k
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = test_set

    def euclideanDistance(self, train_data, test_data, length):
        # Similarity measure
        distance = 0
        # calculate the distance between each feature
        distance = np.linalg.norm(train_data - test_data)
        return distance

    def mahalanobisDistance(self, train_data, test_data, length):
        # Mahalanobis distance is the distance between a point and a distribution
        mean = np.mean(train_x)
        cov_inverse = np.linalg.inv(self.train_covariance)
        diff = test_data - mean
        diff_T = diff.T
        dotproduct = np.dot(np.dot(diff, cov_inverse), diff_T)
        return np.diag(dotproduct)

    def calcEig(self):
        self.train_covariance = np.cov(self.train_x.T)
        self.test_covariance = np.cov(self.test_x.T)
        self.train_eigval, self.train_eigvec = np.linalg.eig(self.train_covariance)
        # self.test_eigval, self.test_eigvec = np.linalg.eig(self.test_covariance)
        # Transpose eigenvectors (see document for np.linalg.eig)
        self.train_eigvec = self.train_eigvec.real.T
        # sortedIdx = np.argsort(self.train_eigval)

    def projection(self):
        # Project MNIST data on N eigenvectors (N-dim eigenspace)
        n = args.eigdim
        train_iter = args.max_train
        test_iter = args.max_test
        self.proj_train_x = []
        self.proj_test_x = []
        self.calcEig() # Calculate eigenvalue and eigenvector (from covariance matrix)

        for i in range(train_iter):
            self.proj_train_x.append(np.dot(self.train_eigvec[:n], self.train_x[i]))
        for i in range(test_iter):
            self.proj_test_x.append(np.dot(self.train_eigvec[:n], self.test_x[i]))
        
        return self.proj_test_x

    def knn(self, test_data):
        # Pick k neighbors and return their instances
        distances = []
        length = test_data.shape[0] # 784
        for i in range(args.max_train):
            if not args.project:
                distance = self.euclideanDistance(self.train_x[i], test_data, length)
            else:
                distance = self.euclideanDistance(self.proj_train_x[i], test_data, length)
            label = self.train_y[i]
            distances.append((label, distance)) # [(idx, distance), ...]
        sorted_dist = sorted(distances, key=lambda tup: tup[1])
        neighbors = []
        for i in range(self.k):
            neighbors.append(sorted_dist[i])
        return neighbors

    def prediction(self, neighbors):
        # Using majority voting
        votes = {}
        for i in range(len(neighbors)):
            pred = neighbors[i][0]
            if pred in votes:
                votes[pred] += 1
            else:
                votes[pred] = 1
        pred = max(votes.keys(), key=lambda key: votes[key])
        return pred

    def accuracy(self, pred_list, test_y):
        correct = 0
        for i in range(args.max_test):
            if pred_list[i] == test_y[i]:
                correct += 1
        acc = (correct / float(args.max_test))
        print('[ # Correct / Total ] : ', '[', correct, '/', args.max_test,']')
        return acc



if __name__ == '__main__':
    train_set, val_set, test_set = load_data('mnist.pkl.gz')
    train_x, train_y = train_set
    val_x, val_y = val_set
    test_x, test_y = test_set

    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape[1])

    classifier = KNN(k=args.k, train_set=train_set, test_set=test_set)

    pred_list = []
    if args.project:
        test_x = classifier.projection()
    for i in range(args.max_test):
        neighbors = classifier.knn(test_x[i])
        pred = classifier.prediction(neighbors)
        if i % 10 == 0:
            print(i, ': ', end='')
            print('[ Prediction: ', pred, '] ', '[ Label: ', test_y[i], ']')
        pred_list.append(pred)

    acc = classifier.accuracy(pred_list, test_y)
    print('[ Accuracy: ', acc, ' ]')
    print('[ k neighbors: ', args.k,' ]')
    print('[ Train samples: ', args.max_train,' ]')
    print('[ Test samples: ', args.max_test,' ]')
    if args.project:
        print('[ N = ', args.eigdim,' dim. eigenspace ]')