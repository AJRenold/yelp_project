#!/usr/bin/env python

"""

P(H|X) = P(X|H) P(H)

input:

    class label (H)
    
    words (X)
    
    Useful:
    P(H) = # of useful reviews / # of total reviews
    P(X|H) = probability word X in a useful review = # of word X in useful review / # of words in useful reviews

    Not useful:
    P(H) = # of not useful reviews / # of total reviews
    P(X|H) = probabilty word X in a not useful review = # of word X in not useful reviews / # of words in not useful reviews

"""
    


import os
from collections import Counter
from collections import defaultdict
from math import log
import re
from itertools import islice
import nltk

class NaiveBayes():

    """
    construct with
    data = list of documents (document is a list of words)
    class_labels = list of document class labels

    len data and class_labels must be equal

    v0 = only 2 class_labels accepted

    """

    def __init__(self,data,class_labels):
        assert len(data) == len(class_labels)
        self.data = data
        self.class_labels = class_labels
        self.labels = self.get_class_labels(class_labels)
        self.stop_words = self.get_stop_words()
        self.train()

    def get_stop_words(self):
        stop_words = os.getcwd() + '/stop-words-english4.txt'
        f = open(stop_words,'r')

        stops = defaultdict(bool)

        for line in f:
            word = re.sub('\s',"",line.lower())
            if word not in stops:
                stops[word] = True
        return stops


    def train(self):
        self.class_desc = self.create_class_descriptions(self.class_labels)
        self.tokenized_records = self.tokenize(self.data)
        self.vocab, self.vocab_count = self.create_vocab(self.tokenized_records, self.class_labels)
        self.vocab_size = self.get_vocab_size(self.vocab)
        self.data_probs = self.create_data_probabilities(self.vocab, self.vocab_count, self.vocab_size)

    def get_class_labels(self, class_labels):
        labels = set(class_labels)
        return list(labels)

    def create_class_descriptions(self,class_labels):
        classes = Counter()
        for item in class_labels:
            classes[str(item)] += 1
            classes['total'] += 1

        labels = [ label for label in classes.keys() if label != 'total' ]

        prob = {}
        for label in labels:
            prob[label] = float(classes[label]) / classes['total']

        print prob

        class_desc = defaultdict(dict)

        for label in labels:
            class_desc[label]['probability'] = prob[label]
            class_desc[label]['count'] = classes[label]

        return class_desc

    def tokenize(self, data):
        tokenized_records = []
        for record in data:
            text = re.sub(r"[\n\.',\!\?\(\)\"\-\*/:]"," ",record)
            words = text.lower().split(" ")

            clean_words = []
            for word in words:
                if word != '' and word != ' ':
                    if word in self.stop_words:
                        pass
                    elif '$' in word:
                        clean_words.append('priceMention')
                    else:
                        clean_words.append(word)

            tokenized_records.append(clean_words)

        return tokenized_records

    def create_vocab(self, tokenized_records, class_labels):
        vocab_count = Counter()
        vocab = defaultdict(Counter)
        for i,record in enumerate(tokenized_records):
            for attr in record:
                vocab[attr][class_labels[i]] += 1
                vocab_count[class_labels[i]] += 1
                vocab_count['total'] += 1

        return self.modify_vocab(vocab), vocab_count

    def word_var(self,word):
        return str('^' + "".join([ l + "+" for l in word ]) + "$")
    
    def modify_vocab(self,vocab):
        
        pattern = self.word_var("love")
        matches = []
        for key in vocab.keys():
            if re.match(pattern,key):
                matches.append([key, vocab[key]])
        #print matches
        return vocab

    def create_data_probabilities(self, vocab, vocab_count, vocab_size):
        labels = self.labels

        prob = defaultdict(defaultdict)
        for label in labels:
            for attr in vocab.keys():

                #print attr, label, data_count[attr][label], class_desc[label]['count']
                #prob[attr][label] = float(data_count[attr][label]) / class_desc[label]['count']
                #print attr, label, float(data_count[attr][label]) / class_count[label]
                prob[attr][label] = (float(vocab[attr][label]) + 1) / ( vocab_count[label] + vocab_size )
                #print attr, prob[attr][label]

        return prob
    
    def get_vocab_size(self, vocab):
        print (len(vocab.keys()))
        return len(vocab.keys())

    def find_max_prob_dif(self):
        data_probs = self.data_probs
        class_desc = self.class_desc
        vocab_count = self.vocab_count
        vocab_size = self.vocab_size
        labels = self.labels

        for word in islice(data_probs.keys(),None):
            probs = []
            for label in labels:
                probs.append([abs(log(data_probs[word][label],10)),label])
            #if probs[0][0] >= 2.6 and probs[1][0] >= 2.6 and probs[0][0] - probs[1][0] < -.25 \
            #        and probs[0][0] < 4.5 and probs[1][0] < 4.5:
            #    print word, probs, (probs[0][0] - probs[1][0])

            if probs[0][0] <= 2:
                print word, probs, (probs[0][0] - probs[1][0])


        print "NOT IN VOCAB 0", abs(log(1.0/(vocab_count['0']+vocab_size),10))
        print "NOT IN VOCAB 1", abs(log(1.0/(vocab_count['1']+vocab_size),10))


    def label_new(self, test_tuple):
        data_probs = self.data_probs
        class_desc = self.class_desc
        vocab_count = self.vocab_count
        vocab_size = self.vocab_size
        labels = self.labels

        probs = []
        test_tuple = self.tokenize([test_tuple])[0]

        """ Removing Proper Nouns with nltk = too slow!!
        pos_remove = ['NNP','NNPS']
        tagged_terms = nltk.pos_tag(test_tuple)
        test_tuple = [ word[0] for word in tagged_terms if word[1] not in pos_remove ]
        """

        for label in labels:
            p = 0
            for attr in test_tuple:
                if attr in data_probs:
                    if data_probs[attr][label] > 0:
                        if abs(log(data_probs[attr][label],10)) > 0:
                            #print label, attr, abs(log(data_probs[attr][label],10))
                            p += abs(log(data_probs[attr][label],10))
                    else:
                        print attr, data_probs[attr][label]

                else:
                    p += abs(log(1.0/ (vocab_count[label] + vocab_size),10))

            probs.append((p + log(class_desc[label]['probability'],10), label))

        probs.sort()
        #print probs
        return probs

        #return probs[0]
#bayes = NaiveBayes([['hello','you','me','run'],['run','sit','jump']],[0,1])
#bayes = NaiveBayes([[],[],[],[]],[0,0,0,1])
#bayes = NaiveBayes([['hello','you','me'],['run','sit','jump']],[0])

if __name__ == "__main__":


    ## TEST FROM BOOK
    data = [[ 'youth','high','no','fair' ], \
        [ 'youth','high','no','excellent' ], \
        [ 'middle_aged','high','no','fair' ], \
        [ 'senior','medium','no','fair' ], \
        [ 'senior','low','yes','fair' ], \
        [ 'senior','low','yes','excellent' ], \
        [ 'middle_aged','low','yes','excellent' ], \
        [ 'youth','medium','no','fair' ], \
        [ 'youth','low','yes','fair' ], \
        [ 'senior','medium','yes','fair' ], \
        [ 'youth','medium','yes','excellent' ], \
        [ 'middle_aged','medium','no','excellent' ], \
        [ 'middle_aged','high','yes','fair' ], \
        [ 'senior','medium','no','excellent' ]]
    class_l = ['no','no','yes','yes','yes','no','yes','no','yes','yes','yes','yes','yes','no']


    b = NaiveBayes(data,class_l)
    new_tuple = ['youth', 'medium', 'yes', 'fair']
    res = b.label_new(new_tuple)
    print round(res[0],3) == 0.028
