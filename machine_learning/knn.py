import math
import sys
import classes

DELTA = 0.000001

class knn:
    """Implementation of k-nearest neighbor classifier"""

    def __init__(self, num_features, k):
        self.k = k
        self.statistics = classes.clazz(num_features,'training_set')
        self.instances = list()
        self.normalized_instances = list()
        self.means = list()
        self.stdevs = list()
        self.neighbors = list()
        self.attempts = 0
        self.hits = 0

    def train(self, instance):
        """Process an instance from a testing set"""
        self.statistics.add_match(instance)
        self.instances.append(instance)

    def normalize_instance(self, instance):
        normalized_instance = list()
        for index,elem in enumerate(instance[:-1]):
            if type(elem) is float:
                normalized_instance.append((elem - self.means[index])/self.stdevs[index])
            else:
                normalized_instance.append(elem)
        normalized_instance.append(instance[-1])
        return normalized_instance

    def finish_training(self):
        for feature in self.statistics.features:
            if feature.is_continuous():
                self.means.append(feature.mean())
                self.stdevs.append(feature.standard_deviation())
            else:
                self.means.append(0)
                self.stdevs.append(1)

        for instance in self.instances:
            self.normalized_instances.append(self.normalize_instance(instance))

    def squared_scalar_distance(self,s1,s2):
        if type(s1) is float and type(s2) is float:
            return math.pow(s1-s2,2)
        elif s1 == s2:
            return 0
        else:
            return 1

    def euclidean_distance(self, i1, i2):
        d = 0
        for index,item in enumerate(i1[0:-1]):
             d += self.squared_scalar_distance(i1[index],i2[index])
        return math.sqrt(d)

    def clear_neighbors(self):
        self.neighbors = list()

    def max_neighbor_distance(self):
        if self.neighbors:
            return max([n.distance for n in self.neighbors])
        else:
            return sys.maxint

    def add_neighbor(self, neighbor):
        if len(self.neighbors) < self.k:
            self.neighbors.append(neighbor)
        else:
            index = 0
            for i,n in enumerate(self.neighbors):
                if self.max_neighbor_distance() - n.distance < DELTA:
                    index = i
            self.neighbors[index] = neighbor

    def classify(self, instance):
        normalized_instance = self.normalize_instance(instance)
        self.attempts += 1
        self.clear_neighbors()

        for cur_inst in self.normalized_instances:
            distance = self.euclidean_distance(cur_inst, normalized_instance)
            if len(self.neighbors) < self.k or distance < self.max_neighbor_distance():
                new_neighbor = classes.neighbor()
                new_neighbor.distance = distance
                new_neighbor.instance = cur_inst
                self.add_neighbor(new_neighbor)

        distribution = dict()
        for neighbor in self.neighbors:
            if not neighbor.instance[-1] in distribution:
                distribution[neighbor.instance[-1]] = 1
            else:
                distribution[neighbor.instance[-1]] += 1

        max = 0
        value = ''

        for item in distribution.items():
            if item[1] > max:
                max = item[1]
                value = item[0]

        if value == instance[-1]:
            self.hits += 1
            print 'Instance was successfully classified as ' + value
        else:
            print 'Instance of class %s was mistakenly classified as %s' %(instance[-1], str(value))
        self.train(instance)
        print 'Hit ratio: ' + str(self.hits/float(self.attempts)) + ' (' + str(self.attempts) + ' attempts)'


