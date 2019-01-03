import unittest
import time
import os
import re
import sparkImplementation as sparkImpl


class TestSparkImplementation(unittest.TestCase):

    articles = 0
    article_count = 0
    start_time = 0
    sc = 0

    test_case_TF_tuples = []
    test_case_IDF_tuples = []
    test_case_TFIDF_tuples = []

    term_frequency_dict = list()
    document_freq = dict()
    tfidf = list()

    @classmethod
    def setUpClass(cls):
        super(cls)
        """
                    Initialize the dataset, in this case the articles RDD
                """
        sc = sparkImpl.sc
        cls.articles = sc.wholeTextFiles("dataset/")
        cls.start_time = time.time()
        cls.article_count = cls.articles.count()

    def test_creatingRDD(self):
        self.assertEqual(50, self.article_count)

    # def setUp(self):

    def test_tokenize(self):
        tokenized_text = self.articles.map(lambda (title, text): (os.path.split(title)[1], sparkImpl.tokenize(text)))
        self.assertEqual(1, 1)

    def test_generateWordCorpus(self):
        tokenized_text = self.articles.map(lambda (title, text): (os.path.split(title)[1], sparkImpl.tokenize(text)))
        word_corpus = tokenized_text.flatMapValues(lambda x: x)
        self.assertEqual(word_corpus.count(), 380)

    def test_generateTF(self):
        tokenized_text = self.articles.map(lambda (title, text): (os.path.split(title)[1], sparkImpl.tokenize(text)))
        word_corpus = tokenized_text.flatMapValues(lambda x: x)
        term_frequency_collection = word_corpus.countByValue()
        self.assertEqual(len(term_frequency_collection.items()), 10)

    def test_generateIDF(self):
        tokenized_text = self.articles.map(lambda (title, text): (os.path.split(title)[1], sparkImpl.tokenize(text)))
        word_corpus = tokenized_text.flatMapValues(lambda x: x)
        idf = word_corpus.distinct().map(lambda (title, word): (word, title)).countByKey()
        self.assertEqual(len(idf.items()), 10)


if __name__ == '__main__':
    unittest.main()
