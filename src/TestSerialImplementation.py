import json
import unittest
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
        if t in tf_list[0]:
            counter += 1
    return counter


class TestSerialImplementation(unittest.TestCase):
    test_case_TF_tuples_in_doc1 = [("limit", 0), ("person", 1), ("four", 0), ("yellow", 1), ("dissuading", 0)]
    test_case_IDF_tuples = [("limit", 6), ("person", 10), ("four", 7), ("yellow", 1), ("dissuading", 2)]
    test_case_TFIDF_tuples = [{'term': 'limit', 'score': 14.55609079175885},
                              {'term': 'person', 'score': 24.141568686511505},
                              {'term': 'four', 'score': 13.621371043387192},
                              {'term': 'yellow', 'score': 3.912023005428146},
                              {'term': 'dissuad', 'score': 6.437751649736401}]
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
        self.term_frequency_dict = serialImpl.computeTFDocument(serialImpl.documents)
        self.assertEqual(50, len(self.term_frequency_dict))
        self.assertEqual(0, check_tf_results(self.test_case_TF_tuples_in_doc1, self.term_frequency_dict))
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
        self.tfidf = serialImpl.compute_tfidf(len(serialImpl.documents), self.term_frequency_dict, self.document_freq)
        self.assertEqual(2845, len(self.tfidf))
        self.assertEqual(5, check_against_results(self.test_case_TFIDF_tuples, self.tfidf))
        print json.dumps(self.tfidf)

        print("TFIDF: " + str(len(self.tfidf)))


if __name__ == '__main__':
    unittest.main()
