import argparse
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
classes = list()

def clear_data(word):
    """Trim garbage from the data entries"""
    return word.strip("{}()\n\r\"\'")

def num_features():
    """Return number of attributes used for classification.
       We assume that all records have all the attributes filled in"""
    return len(training_set[0]) - 1

def read_file(file):
    """Read training data into memory"""

    with open(file, 'r') as a_file:
        for a_line in a_file:
            #stop reading when reached empty line
            if len(a_line) < 2:
                return

            str = a_line.split(",")

            #forming list of all classes
            if clear_data(str[-1]) not in classes:
                classes.append(clear_data(str[-1]))

            #parsing input
            instance = list()
            for word in str:
                value = clear_data(word)
                try:
                    value = float(value)
                except ValueError:
                    pass
                instance.append(value)
            training_set.append(instance)


def split_sets(percentage):
    """Shuffle the training set and move (100-percentage) items to testing set"""
    if percentage == 100:
        return

    global training_set
    global testing_set

    random.shuffle(training_set)
    items_to_move = len(training_set) / 100 * (100 - percentage)
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

#Entry point
if __name__ == '__main__':
    args = parse_args()
    read_file(DATA_SETS_DIR + '/' + args.dataset + '/' + args.dataset + '.data')
    split_sets(args.percentage)

    if args.classifier == 'bayes':
        classifier = bayes.bayes()
        for instance in classes:
            clazz_ = clazz(num_features(),instance)
            classifier.add_class(clazz_)
    elif args.classifier == 'knn':
        classifier = knn.knn(num_features(),args.kvalue)
    else:
         raise RuntimeError

    for instance in training_set:
        classifier.train(instance)

    classifier.finish_training()

    for instance in testing_set:
        classifier.classify(instance)



