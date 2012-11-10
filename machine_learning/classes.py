import math

SAMPLE_CORRECTION = 1.0e-20

class Feature:
    """Contains a list of values for a certain feature and performs math operations on it"""

    def __init__(self):
        self.instances = list()


    def add_instance(self, value):
        """Add a value of an attribute to the training set. Used in classifier's train function."""
        self.instances.append(value)


    def is_continuous(self):
        """Return True if the feature has only continuous instances, false otherwise.
           Is needed because probabilities are computed differently for floats and strings"""

        #For some reason this reduce does not work, use plain loop instead
        #return reduce(lambda x,y: type(x) is float and type(y) is float, self.instances)

        for instance in self.instances:
            if not type(instance) is float:
                return False
        return True


    def mean(self):
        """Return mean for the available instances, works only for continuous features"""

        #We actually can return mean=0 for discrete features, but raising an error seems more logical.
        if not self.is_continuous():
            raise RuntimeError

        #can be substituted with sum(self.values) / len(self.values), but we're not gonna make it that easy
        return reduce(lambda x,y: x + y, self.instances) / len(self.instances)


    def squared_deviation(self,value,mean):
        """Return squared deviation for a supplied continuous value and a mean.
           We do not calculate mean here because of performance"""
        return math.pow(mean-value,2)


    def standard_deviation(self):
        """Return standard deviation for the given set of instances"""
        return math.sqrt(self.variance())


    def variance(self):
        """Return variance for the available instances, works only for continuous features"""

        #We actually can return variance=1 for discrete features, but raising an error seems more logical.
        if not self.is_continuous():
            raise RuntimeError

        #performance optimization
        mean = self.mean()

        #For some reason this reduce does not work, use plain old school loop instead.
        # return reduce(lambda x,y: self.squared_deviation(x, mean) + self.squared_deviation(y, mean), self.instances) / \
        #        len(self.instances)

        stdev = 0
        for instance in self.instances:
            stdev += self.squared_deviation(instance, mean)

        return stdev/float(len(self.instances))


    def prior_probability(self, instance, num_matches):
        """Calculate the prior Bayesian probability of a value for a given outcome"""

        if not self.is_continuous():
            #straightforward computation for string values
            return len(filter(lambda x: x == instance, self.instances)) / float(num_matches) + SAMPLE_CORRECTION

        else:
            #just performance
            variance = self.variance()

            #For continuous features, normal distribution is assumed and probability is computed with
            #normal distribution density function
            return (1/math.sqrt(2*math.pi*variance*variance))*math.exp(-math.pow((instance-self.mean())/variance,2)/2)


class Clazz:
    """An outcome container to keep track of the instances belonging to this certain category"""

    def __init__(self, num_features, value):
        self.num_instances = 0
        self.num_matches = 0
        self.features = list()
        self.value = value
        [self.features.append(Feature()) for _ in range(0,num_features)]


    def add_match(self, instances):
        """Add matching record to the outcome"""
        self.num_instances += 1
        self.num_matches += 1
        [feature.add_instance(instances[index]) for index,feature in enumerate(self.features)]


    def add_instance(self):
        """Add non-matching record to the outcome"""
        self.num_instances += 1


    def likelihood_probability(self):
        """Return likelihood probability"""
        if self.num_instances:
            return float(self.num_matches)/self.num_instances
        else:
            return 0


    def class_probability(self, instances):
        """Calculate Bayesian probability for the outcome based on the training data"""
        return self.likelihood_probability() * reduce(lambda x,y: x * y,
            [feature.prior_probability(instances[index],self.num_matches)
                for index,feature in enumerate(self.features)])





