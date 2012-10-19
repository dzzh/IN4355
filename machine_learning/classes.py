import math

SAMPLE_CORRECTION = 1.0e-20

class feature:
    def __init__(self):
        self.instances = list()

    def add_instance(self, value):
        """Used in classifier's train function to add a value of an attribute to the training set"""
        self.instances.append(value)

    def is_continuous(self):
        """Return True if the feature has only continuous instances, false otherwise.
           Is needed because probabilities are computed differently for floats and strings"""
        return reduce(lambda x,y: type(x) is float and type(y) is float, self.instances)

    def mean(self):
        """Return mean for the available instances, works only for continuous features"""
        if not self.is_continuous():
            raise RuntimeError

        #can be substituted with sum(self.values) / len(self.values), but we're not gonna make it that easy
        return reduce(lambda x,y: x + y, self.instances) / len(self.instances)

    def squared_deviation(self,value,mean):
        """Return squared deviation for a supplied continuous value and a mean.
           We do not calculate mean here because of performance"""
        return math.pow(mean-value,2)

    def variance(self):
        """Return variance for the available instances, works only for continuous features"""
        if not self.is_continuous():
            raise RuntimeError

        _mean = self.mean()
        return reduce(lambda x,y: self.squared_deviation(x, _mean) + self.squared_deviation(y, _mean), self.instances) / \
               len(self.instances)

    def prior_probability(self, instance, num_matches):
        """Calculate the prior probability of a value for a given outcome"""

        if not self.is_continuous():
            #straightforward computation for string values
            return len(filter(lambda x: x == instance, self.instances)) / float(num_matches) + SAMPLE_CORRECTION

        else:
            #just performance
            variance = self.variance()

            #mean-variance formula for continuous attributes
            return (1/math.sqrt(2*math.pi*variance))*math.exp(-math.pow(instance-self.mean(),2)/2*variance)

class clazz:

    def __init__(self, num_features, value):
        self.num_instances = 0
        self.num_matches = 0
        self.features = list()
        self.value = value
        for _ in range(1,num_features):
            self.features.append(feature())

    def add_match(self, instances):
        """Add matching record to the outcome"""
        self.num_instances += 1
        self.num_matches += 1
        for index,feature in enumerate(self.features):
            feature.add_instance(instances[index])

    def add_instance(self):
        """Add non-matching record to the outcome"""
        self.num_instances += 1

    def likelihood_probability(self):
        """Return likelihood probability"""
        return float(self.num_matches)/self.num_instances

    def class_probability(self, instances):
        """Calculate Bayesian probability for the outcome based on the training data"""
        return self.likelihood_probability() * reduce(lambda x,y: x * y,
            [feature.prior_probability(instances[index],self.num_matches) for index,feature in enumerate(self.features)])

class bayes:

    def __init__(self):
        self.classes = list()
        self.attempts = 0
        self.hits = 0

    def add_class(self, outcome):
        """Add a possible outcome"""
        self.classes.append(outcome)

    def train(self, instance):
        """Process a record from a testing set"""
        for clazz in self.classes:
            if clazz.value == instance[-1]:
                clazz.add_match(instance[0:-1])
            else:
                clazz.add_instance()

    def classify(self, instance):
        """Classify a record from a testing set and show results"""

        #print 'Classifying record ' + str(record)
        self.attempts += 1
        argmax = 0
        class_value = ''
        for clazz in self.classes:
            probability = clazz.class_probability(instance[0:-1])
            #print 'Probability for outcome ' + class_.value + ' is ' + str(probability)
            if probability > argmax:
                argmax = probability
                class_value = clazz.value
        if class_value == instance[-1]:
            self.hits += 1
            print 'Instance was successfully classified as ' + class_value
        else:
            print 'Instance of class ' + instance[-1] + ' was mistakenly classified as ' + class_value
        print 'Hit ratio: ' + str(self.hits/float(self.attempts)) + ' (' + str(self.attempts) + ' attempts)'


