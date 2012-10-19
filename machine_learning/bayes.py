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
        """Process a record from a testing set"""
        for clazz in self.classes:
            if clazz.value == instance[-1]:
                clazz.add_match(instance[0:-1])
            else:
                clazz.add_instance()

    def finish_training(self):
        """Not used for Bayes classifier"""
        pass

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