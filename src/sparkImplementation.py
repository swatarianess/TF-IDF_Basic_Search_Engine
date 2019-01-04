import os
import re
import time
import json
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

import numpy as np
from pyspark import SparkContext, SparkConf

stopWords = set(stopwords.words('english'))
stopWords.add("")

# conf = SparkConf().setMaster("local[*]").setAppName("TFIDF Query Search")
# sc = SparkContext(conf=conf)

sc = SparkContext(master="local[8]")
# sc.setLogLevel('INFO')
ps = SnowballStemmer("english")


def tokenize(s):
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
            tf_idf_score = float(value) * np.log(number_of_documents / document_frequency_of_term)

        results.append({"doc": doc, "score": tf_idf_score, "term": term})
    return results


def parallel_tfidf(document_count, term_frequency, document_frequency):
    print("hi")


def query(query_text, tfidf, top_result_count):
    tokens = sc.parallelize(stem(tokenize(query_text))).map(lambda x: (x, 1)).collectAsMap()
    broadcast_tokens = sc.broadcast(tokens)

    joined_tfidf = tfidf \
        .map(lambda (k, v): (k, broadcast_tokens.value.get(k, '-'), v)) \
        .filter(lambda (a, b, c): b != '-')

    s_count = joined_tfidf \
        .map(lambda a: a[2]) \
        .aggregateByKey((0, 0), (lambda (acc, value): (acc[0] + value, acc[1] + 1)), (lambda (acc1, acc2): (acc1[0] + acc2[0], acc1[1] + acc2[1])))

    return s_count.map(lambda (k, v): (v[0] * v[1] / len(tokens), k)).top(top_result_count)


if __name__ == "__main__":
    # create Spark context with Spark configuration
    # sc.setLogLevel("INFO")

    # read input text files present in the directory to RDD
    articles = sc.wholeTextFiles("dataset/")

    start_time = time.time()
    # collect the RDD to a list
    article_count = articles.count()
    # print("Documents Count: " + str(article_count))

    tokenized_text = articles.map(lambda (title, text): (os.path.split(title)[1], tokenize(text)))

    word_corpus = tokenized_text.flatMapValues(lambda x: x).map(lambda (title, text): (title, ps.stem(text)))

    # print "Word corpus: " + str(len(filter(lambda s: (s not in stopWords), word_corpus.values().collect())))
    # print "Word corpus (distinct): " + str(len((filter(lambda s: s not in stopWords, word_corpus.values().distinct().collect()))))
    # print ""

    # with open('resource/spark_word_corpus.json', 'w') as outfile:
    #     json.dump(word_corpus.values().distinct().collect(), outfile, indent=4, sort_keys=True)
    #     print "Word Corpus saved to: " + os.path.realpath(outfile.name)
    #
    # print ""
    # Calculate Term frequencies
    term_frequencyRDD = word_corpus.countByValue()

    # with open('resource/spark_tf.json', 'w') as outfile:
    #     json.dump(word_corpus.countByValue().items(), outfile, indent=4, sort_keys=True)
    #     print "TF saved to: " + os.path.realpath(outfile.name)

    # print("TF Sample(5): " + str(term_frequencyRDD.items()[:5]))

    # Document frequency RDD
    # Gets tokenized corpus, gets distinct values, swaps key and value, counts key appearance
    document_frequencyRDD = word_corpus \
        .distinct() \
        .map(lambda (title, word): (word, title)) \
        .countByKey()

    # print("IDF sample(5): " + str(document_frequencyRDD.items()[:5]))

    # TFIDF RDD
    tf_idf_output = tf_idf(article_count, term_frequencyRDD, document_frequencyRDD)
    print('TF_IDF sample(5)' + str(tf_idf_output[:5]))

    print("Elapsed time: %.4fs" % (time.time() - start_time))
    # TF-IDF scores of Corpus
    tfidf_rdd = sc.parallelize(tf_idf_output).map(lambda x: (x['term'], (x['doc'], x['score'])))
    print "TFIDF_RDD COUNT: " + str(tfidf_rdd.count())
    # print json.dumps(query("Ink helps drive democracy in Asia", tfidf_rdd, 5).count())
