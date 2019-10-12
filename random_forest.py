'''
Using Random Forest algorithm (using sklearn) to classify MNIST dataset
'''
from mnist_data_python3 import load_data
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    train_set, _, test_set = load_data('mnist.pkl.gz')
    train_x, train_y = train_set
    test_x, test_y = test_set
    model = RandomForestClassifier(n_estimators=100)
    print('[ Training... ]')
    model.fit(train_x, train_y)
    print('[ Accuracy: ', model.score(test_x, test_y), ' ]')
    print('Model training complete.')
