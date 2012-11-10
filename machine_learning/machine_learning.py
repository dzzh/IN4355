import argparse
from functools import partial
import os
import random
import bayes
import knn

#Constants
from classes import Clazz
import utils

DATA_SETS_DIR = 'data_sets'
PERCENTAGE_TESTING = 90

#All the discrete classes
cls = list()

#Raw representation of a training set for normalization
statistics = None
means = list()
stdevs = list()

def clear_data(word):
    """Trim garbage from the data entries"""
    return word.strip("{}()\n\r\"\'")


def read_file(file, is_training):
    """Read data set, training or testing"""

    result = list()
    with open(file, 'r') as a_file:
        for a_line in a_file:
            #stop reading when reached empty line
            if len(a_line) < 2:
                break

            str = a_line.split(",")
            #forming list of all classes
            if is_training:
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
            result.append(instance)
    return result


def get_split_offset(percentage, training_set):
    """Shuffle the training set and find offset to move (100-percentage) percents items to testing set"""
    if percentage == 100:
        return

    random.shuffle(training_set)
    items_to_move = int(round(len(training_set) / 100.0 * (100 - percentage)))
    offset = len(training_set) - items_to_move
    return offset


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


def normalize(instances):
    """Perform z-score normalization for continuous features"""
    result = []
    for feature in statistics.features:
        if feature.is_continuous():
            means.append(feature.mean())
            stdevs.append(feature.standard_deviation())
        else:
            means.append(0)
            stdevs.append(1)

    [result.append(normalize_instance(instance)) for instance in instances]
    return result


#Entry point
if __name__ == '__main__':
    args = parse_args()
    file_prefix = DATA_SETS_DIR + '/' + args.dataset + '/' + args.dataset
    read_training = partial(read_file, file=file_prefix+'.data', is_training=True)
    read_testing = partial(read_file, file=file_prefix+'.test', is_training=False)

    #Read training and testing data into variables
    training_set = read_training()
    testing_set = list()
    if os.path.exists(file_prefix + '.test'):
        testing_set = read_testing()
        print 'Testing set is read from a .test file'
    else:
        offset = get_split_offset(args.percentage, training_set)
        testing_set = training_set[offset:]
        training_set = training_set[:offset]
        print 'Testing set is derived from training set'
    print 'Training set: %d instances, testing set: %d instances' %(len(training_set), len(testing_set))

    #Choose classifier
    if args.classifier == 'bayes':
        classifier = bayes.bayes()
        for instance in cls:
            clazz_ = Clazz(len(training_set[0])-1,instance)
            classifier.add_class(clazz_)
    elif args.classifier == 'knn':
        classifier = knn.knn(args.kvalue)
    else:
         raise RuntimeError

    #Normalize continuous features
    statistics = Clazz(len(training_set[0])-1,'training_set')
    [statistics.add_match(instance) for instance in training_set]

    #Train and classify
    [classifier.train(instance) for instance in normalize(training_set)]
    [classifier.classify(normalize_instance(instance)) for instance in testing_set]
    utils.print_results(classifier.attempts,classifier.hits)
    print 'Classification completed'



