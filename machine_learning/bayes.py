import utils

class bayes:
    """Implementation of naive Bayes classifier"""

    def __init__(self):
        self.classes = list()
        self.attempts = 0
        self.hits = 0


    def add_class(self, clazz):
        """Add a possible outcome"""
        self.classes.append(clazz)


    def train(self, instance):
        """Process a record from a training set"""
        for clazz in self.classes:
            if clazz.value == instance[-1]:
                clazz.add_match(instance[0:-1])
            else:
                clazz.add_instance()


    def classify(self, instance):
        """Classify a record from a testing set and show results"""

        self.attempts += 1
        max_probability = 0
        class_value = ''
        for clazz in self.classes:
            probability = clazz.class_probability(instance[0:-1])
            #print 'Probability for outcome ' + class_.value + ' is ' + str(probability)
            if probability > max_probability:
                max_probability = probability
                class_value = clazz.value

        if class_value == instance[-1]:
            self.hits += 1

        if not self.attempts % 100:
            utils.print_results(self.attempts,self.hits)