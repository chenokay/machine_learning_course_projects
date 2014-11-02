"""
This program is an implementation of naive Bayes algorithm.

Gaussian distribution is used to model continuous attributes.

M-estimates is used to smooth the probabilities estimates for discrete
attributes.

Usages:
python nbayes.py m

where m is the value for the m-estimate.
"""

from mldata import *
from random import shuffle, seed
from collections import namedtuple
import optparse
import numpy as np
from scipy.optimize import fmin_bfgs

def command_line_parser():
    """
    Parse command line

    Return a tuple of filename and m
    """
    usage = 'usage: %prog [options] filename'
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('-l', dest='l', type='float',
                      help=("Set the lambda value for penalty when minimize \
                      the conditional log likelihood."))
    opts, args = parser.parse_args()
    file_name = sys.argv[-1]
    assert opts.l >= 0, "lambda should be nonnegative"
    l = opts.l
    return (file_name, l)

def data_parser(dataset, path="."):
    """ Return the example data and schema from the data set """
    data_set = parse_c45(dataset, rootdir=path)
    return (data_set.examples, data_set.schema)

def categorical_to_numeric(data, schema):
    """ Figure out all the possible values of each nominal attribute
        in the schema, map each of those possible values to an integer
        between 0 and k (k being the number of possible values for it),
        and re-encode the data using that mapping (returned). """
    attr_types = map(lambda attr: attr.tup[1], schema)
    nominal_indices = [i for i in range(len(attr_types)) if attr_types[i] is "NOMINAL"]
    for index in nominal_indices:
        values, i = {}, 0
        #  Map each value of the attribute to a number from 0 to k
        for value in schema[index].tup[2]:  # For each of the possible values of the attribute
            if value not in values:
                values[value] = i
                i += 1
        for example in data:
            example[index] = values[example[index]]
    return data

def mean_normalized(data):
    """ For each feature xij in example xi, replace with (xij - mean_j)/std_j """
    for feature in range(1, len(data[0]) - 1):  # Ignore id and class (0 and last)
        values = [example[feature] for example in data]
        std = np.std(values)
        mean = np.mean(values)
        for example in data:
            example[feature] = (example[feature] - mean)/std
    return data

def stratified_cross_validation(data, num_folds):
    """ Generator function that yields a Fold tuple on each iteration,
        which has train (80%) and test (20%) of a random shuffle of data,
        both stratified.  Fold f has f.test and f.train """
    Fold = namedtuple('Fold', 'train test')

    true_indices = [i for i in range(len(data)) if data[i][-1] == True]
    false_indices = [i for i in range(len(data)) if data[i][-1] == False]
    shuffle(true_indices)
    shuffle(false_indices)

    # i.e. if 100 true examples, 5 folds, then [0, 20, 40, 60, 80, 99]
    # Then we can just say i.e. train_cutoffs[4:5] is [80:99], the 5th fold.
    true_cutoffs = [(len(true_indices) / num_folds) * i for i in range(num_folds)]
    false_cutoffs = [(len(false_indices) / num_folds) * i for i in range(num_folds)]
    true_cutoffs.append(len(true_indices) - 1)
    false_cutoffs.append(len(false_indices) - 1)

    for i in range(num_folds):
        # i.e. if 100 trues, then true_cutoffs[1] would be 20, so get indices from 20:40
        # So we're getting like 1/5th of the true examples and 1/5th of the false examples
        # Just work with their indices to go faster.
        test_true_indices = true_indices[true_cutoffs[i]:true_cutoffs[i + 1]]
        test_false_indices = false_indices[false_cutoffs[i]:false_cutoffs[i + 1]]
        train_true_indices = list(set(true_indices) - set(test_true_indices))
        train_false_indices = list(set(false_indices) - set(test_false_indices))

        train_indices = train_true_indices + train_false_indices
        test_indices = test_true_indices + test_false_indices
        shuffle(train_indices)
        shuffle(test_indices)

        # Get the examples corresponding to these indices
        train = [data[j] for j in train_indices]
        test = [data[j] for j in test_indices]
        f = Fold(train=train, test=test)
        yield f

def metrics(y, h):
    """ Returns (accuracy, precision, recall) metrics.
        y is the list of actual truth values.
        h is the list of predicted truth values. """
    tp_tn = 0
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(y)):
        if y[i] == h[i]:
            tp_tn += 1
        if y[i] == True:
            if h[i] == True:
                tp += 1
            else:
                fn += 1
        if y[i] == False and h[i] == True:
            fp += 1
    accuracy = float(tp_tn) / len(y)
    try:
        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)
    except ZeroDivisionError:
        print("Your classifier output all True or all False, try varying the parameters.")
        exit(0)
    return (accuracy, precision, recall)

def area_of_ROC(y, y_confidence):
    """
    :param y: a list of labels
    :param y_confidence:  a list of confidences
    :return: area of ROC
    ROC is the curve with x-axis as false positive rate and y-axis as true
    positive rate. By changing the threshold we get a curve.
    """
    fp_rate_tp_rate = []
    thresholds = [float(i)/10 for i in range(11)]
    for threshold in thresholds:
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        y_hat = []
        for each in y_confidence:
            y_hat.append(each>threshold)
        for i in range(len(y)):
            if y[i] == False:
                if y_hat[i] == True:
                    fp += 1
                else:
                    tn += 1
            else:
                if y_hat[i] == True:
                    tp += 1
                else:
                    fn += 1
        try:
            fp_rate = float(fp) / (fp + tn)
            tp_rate = float(tp) / (tp + fn)
        except ZeroDivisionError:
            print("Your classifier output all True or all False, try varying the parameters.")
            exit(0)
        fp_rate_tp_rate.append((fp_rate, tp_rate))
    area = 0
    for i in range(len(fp_rate_tp_rate)-1):
        area += ((fp_rate_tp_rate[i][0] - fp_rate_tp_rate[i+1][0]) * 0.5 *
                 (fp_rate_tp_rate[i][1] + fp_rate_tp_rate[i+1][1]))
    return area

def output(accuracies, precisions, recalls, auc=None):
    """Generate formatted output"""
    output1 = "Accuracy: {} {}".format(np.mean(accuracies),np.std(accuracies))
    output2 = "Precision: {} {}".format(np.mean(precisions),np.std(precisions))
    output3 = "Recall: {} {}".format(np.mean(recalls),np.std(recalls))
    output4 = "Area under ROC: {}".format(auc)
    print(output1)
    print(output2)
    print(output3)
    print(output4)

class logreg():
    """ Class representation of naive Bayes"""

    def __init__(self, data, schema, l):
        """
        X is data features
        m is the number of data
        n is the size of data features
        y is class label
        l is regularization parameter lambda
        theta is the logistic regression parameters,
          initialized with random numbers ranging from [-0.1, 0.1)
        step is the step size to update theta
        epsilon is the threshold to stop the iteration of updating theta
        """
        self.X = np.array([exp[1:-1] for exp in data]) #[m,n]
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]
        self.y = np.array([exp[-1] for exp in data]).reshape((self.m,1)) #[m,1]
        self.l = l
        self.X = np.append(np.ones((self.m,1),dtype=float),self.X,axis=1)
        # Add a column of ones before X, X -> [m, n+1]
        self.theta = np.random.random((self.n+1, 1))/5-0.1
        # Random initialize the theta in range of [-0.1, 0.1)
        self.step = 0.1
        self.epsilon = 0.001

    def sigmoid(self, x):
        """
        :return: sigmoid(x)
        """
        return 1 / (1 + np.exp(-x))

    def cost_function_reg(self, X, y, theta):
        """
        cost_function_reg method is to calculate to regularized cost function
        and gradient of the cost.
        :return: cost and partial derivatives as grads
        """
        h = self.sigmoid(X.dot(theta)) # [m, 1]
        theta_r = theta[1:] # [n, 1]
        J = (1.0 / self.m) * ((-y.T.dot(np.log(h))) - \
            ((1 - y).T.dot(np.log(1.0 - h)))) \
            + (self.l / (2.0 * self.m)) * (theta_r.T.dot(theta_r)) # [1, 1]
        delta = h - y # [m, 1]
        l_theta_r = np.insert(theta_r, 0, 0).reshape((self.n+1,1)) # [n+1, 1]
        grad = (1.0 / self.m) * X.T.dot(delta) + \
            self.l / self.m * l_theta_r # [n+1, 1]
        return J, grad

    def train(self):
        """
        train method use batch gradient descent to update theta.
        :return:
        """
        theta = self.theta
        J_pre = float('inf')
        i = 0
        while(True):
            (J, grad) = self.cost_function_reg(self.X, self.y, theta)
            if abs(J - J_pre) < 0.001*J:
                break
            else:
                theta -= self.step * grad
                i += 1
                J_pre = J
        self.theta = theta

    def classify(self, data):
        """
        classify method classify the data based on the trained logistic
        regression parameters.
        data is the data to be classified.
        :return: A list of tuple, the tuple is a combination of label and
                 confidence.
        :return example:
            [(True, 0.6), (False, 0.9)]
        """
        data = np.array([exp[1:-1] for exp in data])
        data = np.append(np.ones((data.shape[0],1),dtype=float),data,axis=1)
        predictions = []
        for exp in data:
            confidence = self.sigmoid(exp.T.dot(self.theta))
            predictions.append((confidence>0.5, confidence))
        return predictions

if __name__ == '__main__':

    DEBUG = True
    seed(12345)
    np.random.seed(12345)
    filename, l = command_line_parser()
    (data_object, schema) = data_parser(filename, filename)
    data = [data.features for data in data_object]
    data = categorical_to_numeric(data, schema)
    data = mean_normalized(data)

    accuracies, precisions, recalls = [], [], []
    y_labels = []
    y_confidences = []

    if not DEBUG:
        lr = logreg(data, schema, l)
        lr.train()
        predictions = lr.classify(data)
        data_label = [exp[-1] for exp in data]
        print(metrics(data_label, predictions))

    for fold in stratified_cross_validation(data, 5):
        lr = logreg(fold.train, schema, l)
        lr.train()
        predictions = lr.classify(fold.test)
        predict_labels = [prediction[0] for prediction in predictions]
        predict_confidence = [prediction[1] for prediction in predictions]
        test_labels = [exp[-1] for exp in fold.test]
        accuracy, precision, recall = metrics(test_labels, predict_labels)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        y_labels += test_labels
        y_confidences += predict_confidence

    auc = area_of_ROC(y_labels, y_confidences)
    output(accuracies, precisions, recalls, auc)

        
        


