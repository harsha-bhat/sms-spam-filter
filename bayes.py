from collections import defaultdict
import math

class BayesClassifier:
    def __init__(self):
        self.word_probs = []

    def train(self, training_set):
        num_spam = len([1 for message in training_set if message['msg_type'] == 'spam'])
        num_ham = len([1 for message in training_set if message['msg_type'] == 'ham'])
        counts = self.word_counts(training_set)
        self.word_probs = self.word_probabilities(counts, num_spam, num_ham)

    def classify(self, testing_set):
        for message in testing_set:
            message['prob_spam'] = self.spam_probability(message)
        return testing_set

    def word_counts(self, messages):
        counts = defaultdict(lambda: {'spam': 0, 'ham': 0})
        for message in messages:
            for word in message['words']:
                counts[word]['spam' if message['msg_type'] == 'spam' else 'ham'] += 1
        return counts

    def word_probabilities(self, counts, total_spam, total_ham, k=0.5):
        word_probs = []
        for word, count in counts.items():
            word_probs.append((word, (count['spam'] + k)/ (total_spam + 2 * k), (count['ham'] + k)/ (total_ham + 2 * k)))
        return word_probs

    def spam_probability(self, message):
        log_prob_spam = log_prob_ham = 0.0
        for word, prob_spam, prob_ham in self.word_probs:
            if word in message['words']:
                log_prob_spam += math.log(prob_spam)
                log_prob_ham += math.log(prob_ham)
            else:
                log_prob_spam += math.log(1 - prob_spam)
                log_prob_ham += math.log(1 - prob_ham)
        prob_spam = math.exp(log_prob_spam)
        prob_ham = math.exp(log_prob_ham)
        return prob_spam / (prob_spam + prob_ham)