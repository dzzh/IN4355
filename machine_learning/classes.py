import math

DELTA = 0.000001
SAMPLE_CORRECTION = 1.0e-20

class attribute:
    def __init__(self):
        self.values = list()

    def add_sample(self, value):
        """Used in classifier's train function to add a value of an attribute to the training set"""
        self.values.append(value)

    def floats_eq(self,x,y):
        """Compare two floats for equality with predefined threshold"""
        return x - y < DELTA

    def is_continuous(self):
        """Return True if the attribute has only continuous values, false otherwise.
           Is needed because probabilities are computed differently for floats and strings"""
        return reduce(lambda x,y: type(x) is float and type(y) is float, self.values)

    def mean(self):
        """Return mean for the available values, works only for continuous attributes"""
        if not self.is_continuous():
            raise RuntimeError

        #can be substituted with sum(self.values) / len(self.values), but we're not gonna make it that easy
        return reduce(lambda x,y: x + y, self.values) / len(self.values)

    def squared_deviation(self,value,mean):
        """Return squared deviation for a supplied continuous value and a mean.
           We do not calculate mean here because of performance"""
        return math.pow(mean-value,2)

    def variance(self):
        """Return variance for the available values, works only for continuous attributes"""
        if not self.is_continuous():
            raise RuntimeError

        _mean = self.mean()
        return reduce(lambda x,y: self.squared_deviation(x, _mean) + self.squared_deviation(y, _mean), self.values) / \
               len(self.values)

    def prior_probability(self, value, num_matches):
        """Calculate the prior probability of a value for a given outcome"""

        if not self.is_continuous():
            #straightforward computation for string values
            return len(filter(lambda x: x == value, self.values)) / float(num_matches) + SAMPLE_CORRECTION

        else:
            #just performance
            variance = self.variance()

            #mean-variance formula for continuous attributes
            return (1/math.sqrt(2*math.pi*variance))*math.exp(-math.pow(value-self.mean(),2)/2*variance)

class outcome:

    def __init__(self, num_attributes, value):
        self.num_instances = 0
        self.num_matches = 0
        self.attributes = list()
        self.value = value
        for _ in range(1,num_attributes):
            self.attributes.append(attribute())

    def add_match(self, values):
        """Add matching record to the outcome"""
        self.num_instances += 1
        self.num_matches += 1
        for index,attribute in enumerate(self.attributes):
            attribute.add_sample(values[index])

    def add_instance(self):
        """Add non-matching record to the outcome"""
        self.num_instances += 1

    def likelihood_probability(self):
        """Return likelihood probability"""
        return float(self.num_matches)/self.num_instances

    def outcome_probability(self, values):
        """Calculate Bayesian probability for the outcome based on the training data"""
        return self.likelihood_probability() * reduce(lambda x,y: x * y,
            [attribute.prior_probability(values[index],self.num_matches) for index,attribute in enumerate(self.attributes)])

class bayes:

    def __init__(self):
        self.classes = list()
        self.attempts = 0
        self.hits = 0

    def add_outcome(self, outcome):
        """Add a possible outcome"""
        self.classes.append(outcome)

    def train(self, record):
        """Process a record from a testing set"""
        for class_ in self.classes:
            if class_.value == record[-1]:
                class_.add_match(record[0:-1])
            else:
                class_.add_instance()

    def classify(self, record):
        """Classify a record from a testing set and show results"""

        #print 'Classifying record ' + str(record)
        self.attempts += 1
        argmax = 0
        outcome = ''
        for class_ in self.classes:
            probability = class_.outcome_probability(record[0:-1])
            #print 'Probability for outcome ' + class_.value + ' is ' + str(probability)
            if probability > argmax:
                argmax = probability
                outcome = class_.value
        if outcome == record[-1]:
            self.hits += 1
            print 'Record was successfully classified as ' + outcome
        else:
            print 'Record of class ' + record[-1] + ' was mistakenly classified as ' + outcome
        print 'Hit ratio: ' + str(self.hits/float(self.attempts)) + ' (' + str(self.attempts) + ' attempts)'


