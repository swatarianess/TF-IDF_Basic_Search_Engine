import re
import os
from argparse import ArgumentParser
from collections import Counter
import json
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import operator
import hashlib
import numpy as np
import time

stopWords = set(stopwords.words('english'))
stopWords.add("")
ps = SnowballStemmer("english")
word_corpus = []


def tokenize(s):
    words = re.split("\W+", s.lower())
    result = []

    for w in words:
        if w in stopWords:
            continue
        else:
            result.append(w)
    return result


def stemWord(s):
    words = s
    result = []

    for w in words:
        word_result = ps.stem(w)
        if word_result is not None:
            result.append(word_result)
            word_corpus.append(word_result)
    return result


def cleanDocuments(document_map):
    result = {}
    for key, value in document_map.items():
        result[key] = stemWord(tokenize(value))
    return result


def computeTF(documents):
    """
        Computes the term frequency of terms in the corpus
    :param documents: A dictionary of the frequency of terms in the whole corpus
    :return: A list of pairs of term frequencies in the form of (term, term_frequency)
    """
    flattened = [val for sublist in documents.values() for val in sublist]
    result = {}

    # Key = document_name, value = array of words
    for key, value in documents.items():
        result[key] = Counter(value).items()

    return Counter(flattened).items()
    # return result


def computeTFDocument(documents):
    # type: (dict) -> list[tuple[str,list[tuple[str,int]]]]
    """

    :param documents: The collection of documents
    :return: Returns dictionary of documents with stemmed words
    """
    result = []
    for document_name in documents:
        result.append((document_name, Counter(documents[document_name]).items()))
    return result


def computeIDF(document_corpus):
    # type: (dict) -> dict
    """
        Computes the inverse document frequency
    :rtype: dict(str, int)
    :param document_corpus: Collection of documents in the corpus
    :return: A dictionary
    """
    # Gets tokenized + stemmed corpus, gets distinct values, swaps key and value, counts key appearance
    result = {}  # type: dict[str,int]

    for a in word_corpus:
        result[a] = 0

    for filename, doc_content in document_corpus.items():
        for corpus_term in word_corpus:
            if corpus_term in doc_content:
                result[corpus_term] += 1

    return result


def compute_tfidf(number_of_documents, term_frequency, document_frequency):
    # type: (int, list, dict) -> list[dict[str,[float,int]]]

    # List[Dict[str, Union[Union[float, int], Any]]]
    """
        Calculates tfidf for the whole corpus
    :rtype: list[dict[str,[float,int]]]
    :param number_of_documents: The number of documents in the corpus
    :param term_frequency: A list of tuples representing term frequencies in the corpus
    :param document_frequency: A list of tuples: [{term, df_score}]
    :return: Returns a list of tuples: [{term: term_name, score: tfidf_score}]
    """

    results = []
    for key, value in term_frequency:
        term = key
        tf_idf_score = 0
        document_frequency_of_term = document_frequency[term]
        if document_frequency_of_term > 0:
            tf_idf_score = float(value) * np.log(number_of_documents / document_frequency_of_term)
        results.append({"score": tf_idf_score, "term": term})
    return results


def compute_tfidfDocuments(number_of_documents, term_frequency, document_frequency):
    # type: (int, list, dict) -> list[dict[str,[float,int]]]

    # List[Dict[str, Union[Union[float, int], Any]]]
    """
        Calculates tfidf for the whole corpus
    :rtype: list[dict[str,[float,int]]]
    :param number_of_documents: The number of documents in the corpus
    :param term_frequency: A list of tuples representing term frequencies in the corpus
    :param document_frequency: A list of tuples: [{term, df_score}]
    :return: Returns a list of tuples: [{term: term_name, score: tfidf_score}]
    """
    results = []
    for key, value in term_frequency:
        for word in value:
            term = word[0]
            term_value = word[1]
            tf_idf_score = 0
            document_frequency_of_term = document_frequency[term]
            if document_frequency_of_term > 0:
                tf_idf_score = term_value * np.log(number_of_documents / document_frequency_of_term)
            results.append({"score": tf_idf_score, "term": term})
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
                    temp_dictionary[filename] = read_document_content(os.path.relpath(path + filename))
            except Exception as e:
                raise e
    else:
        print("Path does not exist, exiting...")
        exit(1)
    return temp_dictionary


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-datasetDir",
                        dest="dataset_directory",
                        help="The directory where the documents are stored")
    args = parser.parse_args()

    if args.dataset_directory is not None:
        directory = args.dataset_directory
    else:
        directory = "dataset/"

    documents = load_documents(directory)
    start_time_total = time.time()
    documents = cleanDocuments(documents)

    with open('resource/stopwords.txt', 'w') as outfile:
        outfile.writelines("\n".join(stopWords))
        print "Documents saved to: " + os.path.realpath(outfile.name)

    with open('resource/serial_documents.json', 'w') as outfile:
        json.dump(documents, outfile, indent=4, sort_keys=True)
        print "Documents saved to: " + os.path.realpath(outfile.name)

    print("Document count: " + str(len(documents)))
    print("Word corpus size: " + str(len(word_corpus)))
    print("")

    # Create and save word corpus
    start_time = time.time()
    word_corpus = list(set(word_corpus))
    print("Word corpus (distinct): " + str(len(word_corpus)))
    print("Corpus directory: " + saveCorpus(word_corpus, 'resource/serial_word_corpus.json'))
    print("Elapsed time (Word corpus): %.4fs" % (time.time() - start_time))
    print ""

    # Calculate Term Frequencies
    start_time = time.time()
    print "Document (top): " + str(documents.items()[:1])
    # term_frequency_dict = computeTF(documents)
    term_frequency_dict = computeTFDocument(documents)
    print("Term Frequency (5 elements): " + str(json.dumps(term_frequency_dict[:5])))
    print("Elapsed time (Term frequency): %.4fs" % (time.time() - start_time))

    with open('resource/serial_tf.json', 'w') as outfile:
        json.dump(term_frequency_dict, outfile, sort_keys=True, indent=4)
        print "TF saved to: " + os.path.realpath(outfile.name)
    print ""

    # Calculate IDF values
    start_time = time.time()
    document_freq = computeIDF(documents)
    print("IDF (001.txt, 5 Elements): " + str(document_freq))
    print("Elapsed time (IDF): %.4fs" % (time.time() - start_time))

    with open('resource/serial_IDF.json', 'w') as outfile:
        json.dump(sorted(document_freq.items(), key=lambda x: x[1], reverse=True), outfile, indent=4)
        print "IDF saved to: " + os.path.realpath(outfile.name)
    print ""

    # Calculate TFIDF values
    start_time = time.time()
    tfidf = compute_tfidfDocuments(len(documents), term_frequency_dict, document_freq)
    print("TFIDF (5 Elements): " + str(tfidf[:5]))
    print("Elapsed time (TFIDF): %.4fs" % (time.time() - start_time))

    with open('resource/serial_TFIDF.json', 'w') as outfile:
        json.dump(tfidf, outfile, indent=4, sort_keys=True)
        print "Documents saved to: " + os.path.realpath(outfile.name)
    print ""

    print("Elapsed time: %.4fs" % (time.time() - start_time_total))