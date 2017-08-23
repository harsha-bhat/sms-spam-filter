import re
from bayes import BayesClassifier

def tokenize(line):
    line_split = line.lower().split()
    msg_type = line_split[0]
    words = re.findall("[a-z0-9']+", " ".join(line_split[1:]))
    return msg_type, set(words)

if __name__ == '__main__':
    with open('data/data.txt', 'r') as f:
        messages = []
        for line in f:
            msg_type, words = tokenize(line)
            messages.append({'msg_type': msg_type, 'words': words})

        training_set = messages[:int(len(messages) * 0.75)]
        testing_set = messages[int(len(messages) * 0.75):]

        bayes = BayesClassifier()
        bayes.train(training_set)
        classified = bayes.classify(testing_set)

        true_positive = len([1 for message in classified if message['msg_type'] == 'spam' and message['prob_spam'] > 0.5])
        false_positive = len([1 for message in classified if message['msg_type'] == 'ham' and message['prob_spam'] > 0.5])
        true_negative = len([1 for message in classified if message['msg_type'] == 'ham' and message['prob_spam'] <= 0.5])
        false_negative = len([1 for message in classified if message['msg_type'] == 'spam' and message['prob_spam'] <= 0.5])

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1_score = (2 * precision * recall) / (precision + recall)

        print("True Positives = %d, False Postives = %d" %(true_positive, false_positive))
        print("True Negatives = %d, False Negatives = %d" %(true_negative, false_negative))
        print("Precision = %.2f%%, Recall = %.2f%%, F1 Score = %.2f%%" %(precision*100, recall*100, f1_score*100))