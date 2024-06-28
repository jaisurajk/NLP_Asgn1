import numpy as np
import pandas as pd
from collections import defaultdict
import math

# You need to build your own model here instead of using well-built python packages such as sklearn

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# You can use the models form sklearn packages to check the performance of your own models

class BinaryClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPredictZero(BinaryClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)


class NaiveBayesClassifier(BinaryClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        self.priors = dict()
        self.likelihoods = defaultdict(dict)
        self.vocab_size = 0
        

    def fit(self, X: np.ndarray, Y: pd.Series):
        """Train model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the number of sentences
        """
        self.vocab_size = len(X[0])
        # Get number of sentences that label as positive and negative   
        counts = Y.value_counts() 
        num_positives, num_negatives = (counts[1], counts[0])

        # Calculate p(1) and p(0) using log-probabilities
        self.priors[1] = math.log(num_positives/len(Y))
        self.priors[0] = math.log(num_negatives/len(Y))
        
        # word counts
        word_counts = {0: 0, 1: 0}
        # Add word counts based on label
        for sentence, label in zip(X, Y.to_list()):
            word_counts[label] += sentence

        # Calculate p(word | class) using log-probabilities and add-1 smooothing
        total_positive_word_count = np.sum(word_counts[1])
        total_negative_word_count = np.sum(word_counts[0])
        for i in range(self.vocab_size):
            total_negative_occurrences = word_counts[0][i]
            total_positive_occurrences = word_counts[1][i]
            # log(p(word | class))
            self.likelihoods[i][0] =  math.log((total_negative_occurrences + 1) / (total_negative_word_count + self.vocab_size))
            self.likelihoods[i][1] =  math.log((total_positive_occurrences + 1) / (total_positive_word_count + self.vocab_size))
    
    def predict(self, X: np.ndarray):
        """predict a class based on given formatted data
        
        Arg:
            X: {array} array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        NOTE: makes the assumption that X only contains words in the vocabulary
        """
     
        labels = np.zeros(len(X))
        cur_sentence = 0
        # Iterate through each word in a sentence then assign label
        for sentence in X:
            sum = {0: self.priors[0], 1: self.priors[1]}
            for word in range(self.vocab_size):
                # Add to sum for each class for every present word
                sum[0] += (self.likelihoods[word][0] * sentence[word])
                sum[1] += (self.likelihoods[word][1] * sentence[word])
            labels[cur_sentence] = 0 if sum[0] > sum[1] else 1
            cur_sentence += 1
        
        return labels
    
    def comp(self, pair):
        return pair[1]


    def top_n(self, n, words:dict):
        """find 10 most distinctly positive words and lowest ratio"""

        bot = list()
        top = list()
        for key in self.likelihoods:
            # bot.append((key, self.likelihoods[key][0]/self.likelihoods[key][1]))
            top.append((key, math.exp(self.likelihoods[key][1])/math.exp(self.likelihoods[key][0])))
        
        # bot.sort(key=self.comp)
        top.sort(key=self.comp)

        # bot = [x for x,y in bot[:n]]
        print(top[:n])
        print(top[-10:])
        bot = [x for x,y in top[-10:]]
        top = [x for x,y in top[:n]]
        print(bot)
        print("BOT")
        for i, key in enumerate(words.keys()):
            if i in bot:
                print(key, i)
        
        print("TOP")
        print(top)
        for i, key in enumerate(words.keys()):
            if i in top:
                print(key, i)



class LogisticRegressionClassifier(BinaryClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self, lam=0):
        self.weights = None
        self.bias = None
        self.threshold = 0.5
        self.learning_rate = 0.1
        self.iterations = 50
        self.lam = lam
    
    def sigmoid(self, x):
        """Calculate y hat = sigmoid(W dot X + b)"""
        return 1 / (1+ np.exp(-(np.dot(self.weights, x) + self.bias)))

    def fit(self, X: np.ndarray, Y: pd.Series):
        """Adjust weights for each feature by a specific amount of iterations"""
        num_reviews, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 1

        # Perform iterations by specific amount of times
        for _ in range(0, self.iterations):
            # Iterate over each review and adjust rate
            for i in range(0, num_reviews):
                predict = self.learning_rate * (Y[i]-self.sigmoid(X[i])) 
                # Update weights taking in to account L2 norm
                self.weights = self.weights + predict * X[i] + (self.lam/num_features) * self.weights
                self.bias = predict


       
    def predict(self, X:np.ndarray):
        """Apply weights to given features to make a prediction"""
        num_reviews, _ = X.shape
        labels = np.zeros(num_reviews)

        for i in range(num_reviews):
            res = self.sigmoid(X[i])
            labels[i] = 1 if res > self.threshold else 0

        return labels

# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
