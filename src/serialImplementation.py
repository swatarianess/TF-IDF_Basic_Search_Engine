import json
import math
import os
import pprint
import re
import time
from argparse import ArgumentParser
from termcolor import colored

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stopWords = set(stopwords.words('english'))
stopWords.add("")
ps = SnowballStemmer("english")
pp = pprint.PrettyPrinter(indent=1, depth=3)

word_corpus = []
postings = dict()  # example { "potato" : ["001.txt", "012.txt", "021.txt", ... ], "another": [...], ... }
idf_corpus = dict()


def tokenize(s):
    # type: (str) -> list
    """
            Removes stop-words, and changes words to lowercase.

    :rtype: list()
    :param s: String containing words in a document
    :return: Returns list of strings
    """
    words = re.split("\W+", s.lower())
    result = []

    for w in words:
        if w in stopWords:
            continue
        else:
            result.append(w)
    return result


def stemWord(s):
    # type: (list) -> list
    words = s
    result = []

    for w in words:
        word_result = ps.stem(w)
        if word_result is not None:
            result.append(word_result)
            word_corpus.append(word_result)
    return result


def cleanDocuments(document_map):
    """
            Tokenizes, and extracts base of words. It also removes stopwords from each document's content.

    :param document_map: Dictionary of documents with their content
    :return: Returns dictionary of documents
    """
    result = {}
    for key, value in document_map.items():
        result[key] = stemWord(tokenize(value))
    return result


def computeTF(documents):
    """
        Calculate the term frequency

    :param documents: The collection of documents
    :return: Returns dictionary of terms with their term-frequency.
    """
    result = {}
    for document_name, document_terms in documents.items():  # Iterate keys
        distinct_terms = set(document_terms)  # Set of distinct terms in a document
        result[document_name] = []  # key = document, value = list of tuple(term,score)
        for term in distinct_terms:
            score = float(document_terms.count(term)) / len(document_terms)  # Normalized TF score
            result[document_name].append((term, score))

    return result


def computeIDF(documents):
    # type: (dict) -> dict
    """
        Computes the inverse document frequency

    :rtype: dict(str, int)
    :param documents: Collection of documents in the corpus
    :return: A dictionary
    """
    # Gets tokenized + stemmed corpus, gets distinct values, swaps key and value, counts key appearance
    result = {}  # type: dict[str,int]

    for a in word_corpus:  # Populate results + postings list
        result[a] = 0
        postings[a] = []

    for document_name, doc_content in documents.items():
        for term in word_corpus:
            if term in doc_content:
                result[term] += 1
                postings[term].append(document_name)

    return result


def compute_tfidfDocuments(number_of_documents, term_frequency, document_frequency):
    # type: (int, dict, dict) -> dict[list]
    """
        Calculates tfidf scores for all distinct terms in the corpus

    :rtype: list[dict[str,[float,int]]]
    :param number_of_documents: The number of documents in the corpus
    :param tf: A list of tuples representing term frequencies in the corpus
    :param document_frequency: A list of tuples: [{term, df_score}]
    :return: Returns a dictionary listed tuples: { "001.txt": [("sector", 1.6094379), ("embassi", 2.302585), ()]}
    """
    results = {}  # type: dict[list]
    for key, value in term_frequency.items():
        for word in value:
            term = word[0]
            tf = word[1]
            tf_idf_score = 0
            document_frequency_of_term = document_frequency[term]
            if document_frequency_of_term > 0:
                idf = math.log10(1.0 + (float(number_of_documents) / document_frequency_of_term))
                tf_idf_score = tf * idf
            if key in results:
                results.get(key).append((term, tf_idf_score))
            else:
                results[key] = [(term, tf_idf_score)]
    return results


def read_document_content(path):
    # type: (str) -> str
    """
        Reads the file from the supplied path

    :rtype: str
    :param path: A path to the document file
    :return: A string containing the content of the document
    """
    with open(path, 'r') as document:
        return document.read()


def saveCorpus(corpus, path):
    corpus_directory = ""
    with open(path, 'w') as outfile:
        json.dump(corpus, outfile)
        corpus_directory = os.path.realpath(outfile.name)

    return corpus_directory


def load_documents(path):
    # type: (str) -> dict
    """
        Creates dictionary from all files in folder

    :param path: The path to the folder containing all .txt files
    :return: Returns a dictionary in the form of dict(str, str)
    """
    temp_dictionary = {}
    if os.path.exists(path):
        for filename in os.listdir(path):
            try:
                if filename.endswith(".txt"):
                    temp_dictionary[filename] = read_document_content(path + "\\" + filename)
            except Exception as e:
                raise e
    else:
        print("Path does not exist, exiting...")
        exit(1)
    return temp_dictionary


def computeTFQuery(query):
    # type: (list) -> dict
    result = {}
    distinct_terms = set(query)

    for term in distinct_terms:
        result[term] = float(query.count(term))
    return result


def computeTFIDFQuery(tf_query, idf_corpus):
    # type: (dict, dict) -> dict
    result = {}
    for word in tf_query:
        score = 0
        if word not in idf_corpus.keys():
            result[word] = 0
        else:
            for term in idf_corpus.keys():
                score = tf_query[word] * idf_corpus[term]
        result[word] = score
    return result


def retrievePotentialDocuments(query):
    """
        Retrieves documents that have
    :param query:
    :return:
    """
    result = {}

    for term in query:
        result[term] = postings.get(term)

    return result


def computeDocumentScore(query_vector, document_vector, relevant_documents):
    result = {}

    for doc_name in relevant_documents:  # Iterate through documents relevant to query
        numerator = 0.0
        denominator_q = 0.0
        denominator_d = 0.0
        for term in document_vector.get(doc_name):  # Iterate through term tuples tf
            if term[0] in query_vector.keys():  # Check term is in query vector too
                numerator += float(query_vector[term[0]] * term[1])  # q_i * d_i
                denominator_d += float(query_vector[term[0]] ** 2)  # q^2
                denominator_q += float(term[1] ** 2)  # d^2
        sub_result = math.sqrt(denominator_d) * math.sqrt(denominator_q)  # Denominator result
        result[doc_name] = float(np.sqrt(numerator)) / sub_result
    return result


def ranking(scores, k_amount):
    """
        Ranks documents based on their scores i ascending order
    :param scores: ...
    :return: List of tuples containing (document_name, rank_score)
    """
    sorted_list = sorted(scores.items(), key=lambda (doc, score): (score, doc))[:k_amount]
    return [i[0] for i in sorted_list]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dataDir",
                        dest="dataset_directory",
                        help="The directory where the documents are stored")
    args = parser.parse_args()

    if args.dataset_directory is not None:
        directory = args.dataset_directory
        print "Dataset directory: " + os.path.realpath(directory)
    else:
        directory = "dataset/"

    documents = load_documents(directory)
    start_time_total = time.time()
    documents = cleanDocuments(documents)

    with open('resource/stopwords.txt', 'w') as outfile:
        outfile.writelines("\n".join(stopWords))
        print colored("Stop-words saved to: " + os.path.realpath(outfile.name), 'cyan')

    with open('resource/serial_documents.json', 'w') as outfile:
        json.dump(documents, outfile, indent=4, sort_keys=True)
        print colored("Documents saved to: " + os.path.realpath(outfile.name), 'cyan')

    print("Document count: " + str(len(documents)))
    print("Word corpus size: " + str(len(word_corpus)))
    print("")

    # Create and save word corpus
    start_time = time.time()
    word_corpus = list(set(word_corpus))
    # print("Word corpus (distinct): " + str(len(word_corpus)))
    # print("Corpus directory: " + saveCorpus(word_corpus, 'resource/serial_word_corpus.json'))
    print colored("Elapsed time (Word corpus): %.4fs" % (time.time() - start_time), 'green')

    #####################################################
    # Calculate Term Frequencies #
    start_time = time.time()
    tf_corpus = computeTF(documents)
    # print("Term Frequency (5 elements): " + str(tf_corpus.items()[:5]))
    print colored("Elapsed time (Term frequency): %.4fs" % (time.time() - start_time), 'green')

    with open('resource/serial_tf.json', 'w') as outfile:
        json.dump(tf_corpus, outfile, sort_keys=True, indent=4)
        print colored("TF saved to: " + os.path.realpath(outfile.name), 'cyan')
    print ""
    #####################################################

    #####################################################
    # Calculate IDF values
    start_time = time.time()
    idf_corpus = computeIDF(documents)
    # print "IDF (001.txt, 5 Elements): " + str(idf_corpus)
    print colored("Elapsed time (IDF): %.4fs" % (time.time() - start_time), 'green')

    with open('resource/serial_IDF.json', 'w') as outfile:
        json.dump(sorted(idf_corpus.items(), key=lambda x: x[1], reverse=True), outfile, indent=4)
        print colored("IDF saved to: " + os.path.realpath(outfile.name), 'cyan')

    with open('resource/serial_Postings.json', 'w') as outfile:
        json.dump(postings, outfile, indent=4)
        print colored("Postings saved to: " + os.path.realpath(outfile.name), 'cyan')
    print""
    #####################################################

    #####################################################
    # Calculate TFIDF values
    start_time = time.time()
    tfidf = compute_tfidfDocuments(len(documents), tf_corpus, idf_corpus)
    # print("TFIDF (5 Elements): " + str(tfidf.items()[:5]))
    print colored("Elapsed time (TFIDF): %.4fs" % (time.time() - start_time), 'green')

    with open('resource/serial_TFIDF.json', 'w') as outfile:
        json.dump(tfidf, outfile, indent=4, sort_keys=True)
        print colored("TFIDF saved to: " + os.path.realpath(outfile.name), 'cyan')
    #####################################################
    print colored("Index time: %.4fs" % (time.time() - start_time_total), 'green')

    while True:
        query = input('Please enter a query: \n')
        raw_query = query
        start_time = time.time()

        query = stemWord(tokenize(str(query)))
        tf_query = computeTFQuery(query)
        tfidf_query = computeTFIDFQuery(tf_query=tf_query, idf_corpus=idf_corpus)
        potential_docs = retrievePotentialDocuments(query)
        if len(potential_docs.values()) > 0 and None not in potential_docs.values():
            relevant_documents = set(x for l in potential_docs.values() for x in l)
            pprint.pprint("Relevant documents: " + str(relevant_documents))
            print ""

            scores = computeDocumentScore(tfidf_query, tfidf, relevant_documents)
            print "Scores:" + str(scores)
            print "Rank:" + str(ranking(scores, 5)) + '\n'
            print colored("Query time: %.4fs" % (time.time() - start_time), 'cyan')
            print ''
        else:
            print colored("No documents for query: " + ''.join(raw_query), 'red')
            print ""
