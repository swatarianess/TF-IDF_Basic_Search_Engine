import json
import os
import unittest

from termcolor import colored

import serialImplementation as serialImpl
from collections import Counter


def check_against_results(tuple_list, collection):
    count = 0
    for item in tuple_list:
        if item in collection:
            count += 1
    return count


def check_tf_results(tuple_list, tf_list):
    counter = 0
    for t in tuple_list:
        if t in tf_list[0]:  # presume that first item in list is 001.txt
            counter += 1
    return counter


class TestSerialImplementation(unittest.TestCase):
    test_case_TF_tuples_in_doc1 = [("limit", 0), ("person", 1), ("four", 0), ("yellow", 1), ("dissuading", 0)]
    test_case_IDF_tuples = [("limit", 6), ("person", 10), ("four", 7), ("yellow", 1), ("dissuading", 2)]
    test_case_TFIDF_tuples = [{'term': 'limit', 'score': 2.0794415416798357},
                              {'term': 'person', 'score': 1.6094379124341003},
                              {'term': 'four', 'score': 1.9459101490553132},
                              {'term': 'yellow', 'score': 3.912023005428146},
                              {'term': 'dissuad', 'score': 3.2188758248682006}]
    term_frequency_dict = list()
    document_freq = dict()
    tfidf = list()

    def setUp(self):
        """
            Initialize the dataset
        """
        serialImpl.documents = serialImpl.cleanDocuments(serialImpl.load_documents("dataset/"))
        serialImpl.word_corpus = list(set(serialImpl.word_corpus))

    def test_documentCount(self):
        self.assertEqual(len(serialImpl.documents), 50)

    def test_tf(self):
        self.term_frequency_dict = serialImpl.computeTF(serialImpl.documents)
        self.assertEqual(50, len(self.term_frequency_dict))

        list_of_terms = self.term_frequency_dict.get("001.txt")
        counter = 0.0
        for tup in list_of_terms:
            counter += tup[1]

        print counter

        print json.dumps(self.term_frequency_dict)
        print("TF Size: " + str(len(self.term_frequency_dict)))

    def test_idf(self):
        self.document_freq = serialImpl.computeIDF(serialImpl.documents)
        self.assertEqual(2845, len(self.document_freq))
        self.assertEqual(4, check_against_results(self.test_case_IDF_tuples, self.document_freq.items()))

        print("IDF: " + str(len(self.document_freq.items())))

    def test_tfidf(self):
        self.term_frequency_dict = serialImpl.computeTF(serialImpl.documents)
        self.document_freq = serialImpl.computeIDF(serialImpl.documents)
        self.tfidf = serialImpl.compute_tfidfDocuments(len(serialImpl.documents), self.term_frequency_dict, self.document_freq)
        self.assertEqual(50, len(self.tfidf))
        # self.assertEqual(5, check_against_results(self.test_case_TFIDF_tuples, self.tfidf))

        with open('resource/test_serial_tfidf.json', 'w') as outfile:
            json.dump(self.tfidf, outfile, indent=4, sort_keys=True)
            print colored("TEST_TFIDF saved to: " + os.path.realpath(outfile.name), 'cyan')

        print("TFIDF: " + str(len(self.tfidf)))

    def test_computeTFQuery(self):
        query = "Ink helps drive democracy in Asia asia"
        clean_query = serialImpl.stemWord(serialImpl.tokenize(query))
        print serialImpl.computeTFQuery(clean_query)

    def test_computeTFIDFQuery(self):
        docs = serialImpl.documents
        self.term_frequency_dict = serialImpl.computeTF(docs)
        self.document_freq = serialImpl.computeIDF(docs)
        self.tfidf = serialImpl.compute_tfidfDocuments(len(serialImpl.documents), self.term_frequency_dict, self.document_freq)

        # print "TFIDF_Query: " + str(serialImpl.computeTFIDFQuery("Ink bananas helps amber drive drive democracy in Asia", self.document_freq))

    def test_computeDotProduct(self):
        docs = serialImpl.documents
        res = serialImpl.computeTF(docs)
        print res


if __name__ == '__main__':
    unittest.main()
