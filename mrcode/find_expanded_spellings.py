#!/usr/bin/env python

from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol
import re

## Currently will run locally on 1000 records, but fails on AWS

def word_var(word):
    return str('^' + "".join([ l + "+" for l in word ]) + "$")

class FindExpandedSpellings(MRJob):
    INPUT_PROTOCOL = JSONValueProtocol

    def extract_words(self, _, record):

        text = record['text']
        for word in text.lower().split(' '):
            word = re.sub('[^A-Za-z ]+',"",word)
            if word != " " and word != "" and len(word) >= 2 and len(word) < 34:
                yield word[0], word

    def create_combos(self,first_letter, words):
        #yield first_letter, list(words)
        word_list = list(words)
        base_words = {}
        seen = {}

        for word in word_list:
            if word not in seen:
                pattern = word_var(word)
                variations = list(set([ test_word for test_word in word_list if re.search(pattern, test_word ) ]))
                variations.sort(key=len, reverse=False)
                for var in variations:
                        seen[var] = True
                if len(variations) >= 2:
                    base_words[variations[0]] = variations

        yield first_letter, base_words

    def map_expanded_to_base_word(self, _, base_words):

        for word in base_words.keys():
            for expanded_word in base_words[word]:
                yield expanded_word, word


    def steps(self):
        
        return [self.mr(self.extract_words, self.create_combos),
                self.mr(mapper=self.map_expanded_to_base_word)]

if __name__ == '__main__':
    FindExpandedSpellings.run()
