import argparse
import os
import random
import bayes
import knn

#Constants
from classes import clazz

DATA_SETS_DIR = 'data_sets'
PERCENTAGE_TESTING = 90

#Dataset-related variables
training_set = list()
testing_set = list()
cls = list()
statistics = None
instances = list()
normalized_instances = list()
means = list()
stdevs = list()

def clear_data(word):
    """Trim garbage from the data entries"""
    return word.strip("{}()\n\r\"\'")

def num_features():
    """Return number of features used for classification.
       We assume that all records have all the attributes filled in"""
    return len(training_set[0]) - 1

def read_file(file, var):
    """Read training data into memory"""

    with open(file, 'r') as a_file:
        for a_line in a_file:
            #stop reading when reached empty line
            if len(a_line) < 2:
                return

            str = a_line.split(",")

            #forming list of all classes
            if clear_data(str[-1]) not in cls:
                cls.append(clear_data(str[-1]))

            #parsing input
            instance = list()
            for word in str:
                value = clear_data(word)
                try:
                    value = float(value)
                except ValueError:
                    pass
                instance.append(value)
            var.append(instance)


def split_sets(percentage):
    """Shuffle the training set and move (100-percentage) percents items to testing set"""
    if percentage == 100:
        return

    global training_set
    global testing_set
    random.shuffle(training_set)
    items_to_move = int(round(len(training_set) / 100.0 * (100 - percentage)))
    offset = len(training_set) - items_to_move
    testing_set = training_set[offset:]
    training_set = training_set[:offset]

def parse_args():
    """Parse command-line args"""

    parser = argparse.ArgumentParser(
        description='Implements a number of machine-learning algorithms.')
    parser.add_argument('dataset', metavar='D', help='dataset to work with')
    parser.add_argument('-c', '--classifier', choices=['bayes', 'knn'],
        default='bayes', help='classifier to use: naive Bayes or k-nearest neighbor')
    parser.add_argument('-p', '--percentage', default=90, type=int,
        help='Percentage of data set that is used for training, the rest is used for testing')
    parser.add_argument('-k', '--kvalue', default=3, type=int,
        help='k-value for k-nearest neighbor classifier')
    return parser.parse_args()

def normalize_instance(instance):
    """Normalize a given instance using z-score"""
    normalized_instance = list()
    for index,elem in enumerate(instance[:-1]):
        if type(elem) is float:
            normalized_instance.append((elem - means[index])/stdevs[index])
        else:
            normalized_instance.append(elem)
    normalized_instance.append(instance[-1])
    return normalized_instance

def normalize():
    """Perform z-score normalization for continuous features"""
    for feature in statistics.features:
        if feature.is_continuous():
            means.append(feature.mean())
            stdevs.append(feature.standard_deviation())
        else:
            means.append(0)
            stdevs.append(1)

    for instance in instances:
        normalized_instances.append(normalize_instance(instance))

#Entry point
if __name__ == '__main__':
    args = parse_args()
    file_prefix = DATA_SETS_DIR + '/' + args.dataset + '/' + args.dataset
    read_file(file_prefix + '.data', training_set)
    #If
    if os.path.exists(file_prefix + '.test'):
        read_file(file_prefix + '.test', testing_set)
        print 'Testing set is read from a .test file'
    else:
        split_sets(args.percentage)
        print 'Testing set is derived from training set'

    print 'Training set: %d instances, testing set: %d instances' %(len(training_set), len(testing_set))

    if args.classifier == 'bayes':
        classifier = bayes.bayes()
        for instance in cls:
            clazz_ = clazz(num_features(),instance)
            classifier.add_class(clazz_)
    elif args.classifier == 'knn':
        classifier = knn.knn(args.kvalue)
    else:
         raise RuntimeError

    #Normalize continuous features
    statistics = clazz(num_features(),'training_set')
    for instance in training_set:
        statistics.add_match(instance)
        instances.append(instance)
    normalize()

    for instance in normalized_instances:
        classifier.train(instance)

    classifier.instances = normalized_instances

    for instance in testing_set:
        classifier.classify(normalize_instance(instance))



