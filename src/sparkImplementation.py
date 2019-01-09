import os
import pprint
import re
import time
import json
from termcolor import colored
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.rdd import PipelinedRDD

stopWords = set(stopwords.words('english'))
stopWords.add("")

# conf = SparkConf().setMaster("local[*]").setAppName("TFIDF Query Search")
# sc = SparkContext(conf=conf)

sc = SparkContext(master="local[8]")
sc.setLogLevel('INFO')
ps = SnowballStemmer("english")


def tokenize(s):
    """
        Tokenizes the words
    :param s:
    :return:
    """
    words = re.split("\W+", s.lower())
    result = []

    for w in words:
        if w in stopWords:
            continue
        else:
            result.append(w)
    return words


def stem(s):
    words = s
    result = []

    for w in words:
        word_result = ps.stem(w)
        if word_result is not None:
            result.append(word_result)
    return result


# parallelism on this
def tf_idf(number_of_documents, term_frequency, document_frequency):
    results = []
    for key, value in term_frequency.items():
        doc = key[0]
        term = key[1]
        tf_idf_score = 0
        document_frequency_of_term = document_frequency[term]
        if document_frequency_of_term > 0:
            tf_idf_score = float(value) * np.log(float(number_of_documents) / document_frequency_of_term)

        results.append({"doc": doc, "score": tf_idf_score, "term": term})
    return results


def parallel_tfidf(number_of_documents, term_frequency, document_frequency):
    print("parallel_tfidf")

    # document_count => articles.count()
    # term_frequency => word_corpus.countByValue()
    # document_frequency => word_corpus.distinct().map(lambda (title, word): (word, title)).countByKey()

    # Iterate through term_frequencies
    # Get document_name
    # Get term
    # set a counter 'tf_idf_score'
    # Get the document_frequency of the term
    # ignore all terms with document_frequency < 0
    # Multiply

    # tf = sc.parallelize(term_frequency)
    idf = sc.parallelize(document_frequency)

    # pprint.pprint(number_of_documents)
    # pprint.pprint(tf.collectAsMap().items())
    # pprint.pprint(idf.collectAsMap().items())
    return 0


def query(query_text, top_result_count):
    tokens = sc.parallelize(stem(tokenize(query_text))).map(lambda x: (x, 1)).collectAsMap()
    broadcast_tokens = sc.broadcast(tokens)

    joined_tfidf = tfidf_rdd \
        .map(lambda (k, v): (k, broadcast_tokens.value.get(k, '-'), v)) \
        .filter(lambda (a, b, c): b != '-')

    scount = joined_tfidf \
        .map(lambda a: a[2]) \
        .aggregateByKey((0, 0), (lambda (acc, value): (acc[0] + value, acc[1] + 1)), (lambda (acc1, acc2): (acc1[0] + acc2[0], acc1[1] + acc2[1])))

    return scount.map(lambda (k, v): (v[0] * v[1] / len(tokens), k)).top(top_result_count)


if __name__ == "__main__":
    # create Spark context with Spark configuration

    # read input text files present in the directory to RDD
    documents = sc.wholeTextFiles("dataset/")

    start_time = time.time()

    # collect the RDD to a list
    article_count = documents.count()

    tokenized_text = documents.map(lambda (title, text): (os.path.split(title)[1], tokenize(text)))

    word_corpus = tokenized_text.flatMapValues(lambda x: x).map(lambda (title, text): (title, ps.stem(text)))

    # with open('resource/spark_word_corpus.json', 'w') as outfile:
    #     json.dump(word_corpus.values().distinct().collect(), outfile, indent=4, sort_keys=True)
    #     print "Word Corpus saved to: " + os.path.realpath(outfile.name)

    # print ""
    # Calculate Term frequencies
    tf_dict = word_corpus.countByValue()

    # with open('resource/spark_tf.json', 'w') as outfile:
    #     json.dump(tf_dict.items(), outfile, indent=4, sort_keys=True)
    #     print "TF saved to: " + os.path.realpath(outfile.name)

    # print("TF Sample(5): " + str(tf_dict.items()[:5]))

    # Document frequency RDD
    # Gets tokenized corpus, gets distinct values, swaps key and value, counts key appearance
    idf_dict = word_corpus \
        .distinct() \
        .map(lambda (title, word): (word, title)) \
        .countByKey()

    # with open('resource/spark_idf.json', 'w') as outfile:
    #     json.dump(idf_dict, outfile, indent=4, sort_keys=True)
    #     print "TF saved to: " + os.path.realpath(outfile.name)

    # print("IDF sample(5): " + str(document_frequencyRDD.items()[:5]))

    # TFIDF RDD
    # tfidf = parallel_tfidf(article_count, tf_dict, idf_dict)

    tf_idf_output = tf_idf(article_count, tf_dict, idf_dict)
    # print('TF_IDF sample(5)' + str(tf_idf_output[:5]))

    print("Elapsed time: %.4fs" % (time.time() - start_time))
    # TF-IDF scores of Corpus

    # tfidf_rdd = sc.parallelize(tf_idf_output).map(lambda x: (x['term'], (x['doc'], x['score'])))
    # print "TFIDF_RDD COUNT: " + str(tfidf_rdd.count())

    # input_query = input('Enter Query: ')
    # query("Ink helps drive democracy in Asia", 5)
