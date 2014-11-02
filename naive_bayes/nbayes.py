"""
This program is an implementation of naive Bayes algorithm.

Gaussian distribution is used to model continuous attributes.

M-estimates is used to smooth the probabilities estimates for discrete
attributes.

Usages:
python nbayes.py -m

where m is the value for the m-estimate.
"""

from mldata import *
from random import shuffle, seed
from collections import namedtuple, defaultdict
import optparse
import math
import numpy as np

DEBUG = True

def command_line_parser():
    """
    Parse command line

    Return a tuple of filename and m
    """
    usage = 'usage: %prog [options] filename'
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('-m', dest='m', type='float',
                      help=("Set the m value for m-estimate."))
    opts, args = parser.parse_args()
    file_name = sys.argv[-1]
    m = opts.m
    return (file_name, m)

def data_parser(dataset, path="."):
    """ Return the example data and schema from the data set """
    data_set = parse_c45(dataset, rootdir=path)
    return (data_set.examples, data_set.schema)

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
        print("The classifier output all True or all False.")
        exit(0)
    return (accuracy, precision, recall)

def area_of_ROC(data):
    """ Returns the AUC given two lists.
        Data is [ [y values] [y confidence values] ] """
    pass

def output(accuracies, precisions, recalls):
    """Generate formatted output"""
    output1 = "Accuracy: {} {}".format(np.mean(accuracies),np.std(accuracies))
    output2 = "Precision: {} {}".format(np.mean(precisions),np.std(precisions))
    output3 = "Recall: {} {}".format(np.mean(recalls),np.std(recalls))
    output4 = "Area under ROC: Not available"
    print(output1)
    print(output2)
    print(output3)
    print(output4)

def gaussian(x, mu, sig, log=False):
    """
    :param x:
    :param mu:
    :param sig:
    :param log: When false, it returns gaussian probability, when True, it
     returns log gaussian probability, to prevent return 0 because of precision.
    :return:
    """
    if not log:
        return 1. / (sig * np.power(2 * math.pi, 0.5)) \
           * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    else:
        return math.log(1. / (sig * np.power(2 * math.pi, 0.5))) - \
            np.power(x - mu, 2.) / (2 * np.power(sig, 2.)) # __Bonus__

class nbayes:
    """ Class representation of naive Bayes"""

    def __init__(self, data, schema, m):
        """
        The dictionary "label_counts" saves each label total counts.
        Example:
        >>> label_counts
        >>> {False: 40, True:60}

        The dictionary "feature_counts" saves the counts of a combination of
        (label, feature_index, feature_value). And the key of dictionary is the
        tuple of (label, feature_index, feature_value).
        Example:
        >>> feature_counts
        >>> {(False, 1, '+'): 2, (False, 1, '-'): 1, (False, 1, '0'): 2}

        The dictionary "nb_parameter" is the Naive Bayes parameter. It saves
        the probability of each parameter for nominal attributes. For
        continuous attributes, it saves the mean and the variance.
        Example for nominal attributes:
        >>> nb_parameter
        >>> {(False, 1, '+'): 0.4, (False, 1, '-'): 0.2, (False, 1, '0'): 0.4}
        Example for continuous attributes:
        >>> nb_parameter
        >>> {(False, 2): (1, 0.1)}

        """
        self.data = data
        self.schema = schema
        self.m_original = m
        self.p = defaultdict(lambda: 0)
        self.feature_len = len(schema)-2
        self.label_counts = defaultdict(lambda: 0)
        self.feature_counts = defaultdict(lambda: 0)
        self.nb_parameter = defaultdict(lambda: 0)

        for feature_idx in range(self.feature_len):
            if schema[feature_idx+1].type != 'CONTINUOUS':
                # For nominal or binary attributes
                self.p[feature_idx] = 1.0 / len(schema[feature_idx+1].values)
                # Initialize prior estimate of the probability
                for value in schema[feature_idx+1].values:
                    for label in (False, True):
                        self.feature_counts[(label, feature_idx, value)] = 0
                        # Initialize all possible feature counts to zero

        if m < 0:
            self.m = {key: 1/value for key, value in self.p.items()}
            # If m < 0, use Lapalce smoothing
        else:
            self.m = {key: m for key in self.p}

    def train(self):
        """
        The train method:
        a. For nominal or binary attributes:
            Firstly counts the label_counts and feature_counts.
            Secondly using Naive Bayes Maximum Likelihood Estimate method
            (proved results) with m-estimate to calculate the Naive Bayes
            parameter dictionary.
        b. For continuous attributes:
            Use MLE to calculate mean and standard deviation for each feature
            combined with each label.

        """
        true_data = []
        false_data = []
        # Record true data and false data for future MLE use for
        # continuous attributes

        for example in self.data:
            self.label_counts[example[-1]] += 1
            if example[-1]:
                true_data.append(example)
            else:
                false_data.append(example)
            for feature_idx in range(self.feature_len):
                if self.schema[feature_idx+1].type != 'CONTINUOUS':
                    self.feature_counts[(example[-1], feature_idx,
                                         example[feature_idx+1])] += 1

        for parameter in self.feature_counts:
            self.nb_parameter[parameter] = (self.feature_counts[parameter] + \
                self.m[parameter[1]] * self.p[parameter[1]]) / \
                    (self.label_counts[parameter[0]] + self.m[parameter[1]])
            # Update nominal attribute parameter



        true_data = np.array(true_data, dtype=object) # __Bonus__
        false_data = np.array(false_data, dtype=object)

        # Convert list of data into numpy array for column query in the future

        if not DEBUG:
            print(np.mean(true_data[:,2][0]))
            return

        for feature_idx in range(self.feature_len):
            if self.schema[feature_idx+1].type == 'CONTINUOUS':
                true_std = np.std(true_data[:,feature_idx+1]) if \
                    np.std(true_data[:,feature_idx+1]) > 0.1 else 0.1
                false_std = np.std(false_data[:,feature_idx+1]) if \
                    np.std(false_data[:,feature_idx+1]) > 0.1 else 0.1
                # Use a minimum standard deviation of 0.1 to prevent
                # excessive influence of this particular attribute

                self.nb_parameter[(True, feature_idx)] = \
                    (np.mean(true_data[:,feature_idx+1]), true_std)
                self.nb_parameter[(False, feature_idx)] = \
                    (np.mean(false_data[:,feature_idx+1]), false_std)
            # Update continuous attribute parameter


    def predict(self, data):
        """
        :param: data (testing data list)
        :return: label (True/False)
        The predict method:
            Calculate p(X=x, Y=True) and p(X=x, Y=False), and return the
            label of the greater one
            When calculate p, log arithmetic is used to avoid multiplying too
            many probability together.
        """
        predictions = []
        for example in data:
            p_true = 0
            p_false = 0
            if self.m_original != 0:
                for feature_idx in range(self.feature_len):
                    if self.schema[feature_idx+1].type != 'CONTINUOUS':
                        p_true += math.log(self.nb_parameter[True, \
                                        feature_idx, example[feature_idx+1]])
                        p_false += math.log(self.nb_parameter[False,
                                        feature_idx, example[feature_idx+1]])
                    else:
                        p_true += gaussian(example[feature_idx+1], \
                            *(self.nb_parameter[True,feature_idx]),log=True)
                        p_false += gaussian(example[feature_idx+1], \
                            *(self.nb_parameter[False,feature_idx]),log=True)
            else:
                for feature_idx in range(self.feature_len):
                    if self.schema[feature_idx+1].type != 'CONTINUOUS':
                        tmp_true = self.nb_parameter[True, feature_idx, \
                                                example[feature_idx+1]]
                        tmp_false = self.nb_parameter[False, feature_idx, \
                                                example[feature_idx+1]]
                        p_true += math.log(tmp_true) if tmp_true != 0 else 0
                        p_false += math.log(tmp_false) if tmp_false != 0 else 0
                    else:
                        p_true += gaussian(example[feature_idx+1], \
                            *(self.nb_parameter[True, feature_idx]),log=True)
                        p_false += gaussian(example[feature_idx+1], \
                            *(self.nb_parameter[False, feature_idx]),log=True)

            predictions.append(p_true > p_false)
        return predictions


if __name__ == '__main__':

    seed(12345)
    filename, m = command_line_parser()
    (data_object, schema) = data_parser(filename, filename)
    data = [data.features for data in data_object]
    accuracies, precisions, recalls = [], [], []

    if not DEBUG:
        print(type(schema[3].type))

    if not DEBUG:
        print(gaussian(-100,0,1))
        print(gaussian(-100,0,1, True))
        exit(0)

    if not DEBUG:
        nb = nbayes(data, schema, m)
        print(nb.p)
        print(nb.m)
        nb.train()
        print(nb.label_counts)
        print(nb.feature_counts)
        print(nb.nb_parameter)
        print(nb.predict(data))

    for fold in stratified_cross_validation(data, 5):
        nb = nbayes(fold.train, schema, m)
        nb.train()
        y_hat = nb.predict(fold.test)
        y = [exp[-1] for exp in fold.test]
        accuracy, precision, recall = metrics(y, y_hat)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

    output(accuracies, precisions, recalls)

        
        


