import csv
import os

import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from pathlib import Path


def loadModel():
    modelFilePath = os.path.dirname(__file__) + "//data/w2v.model"
    model = Word2Vec.load(modelFilePath)
    list = [["<SOS>", "<EOS>", "<PAD>", "<UNK>"]]
    model.build_vocab(list, update=True)
    return model


def writeReportToCSV(dictionaryData, filename):
    print('dirname:     ', os.path.dirname(__file__))
    folderPath = os.path.dirname(__file__) + "//data"
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    datafilePath = Path(folderPath + "//" + filename)

    fieldNames = list(dictionaryData[0].keys())

    with open(datafilePath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldNames)
        writer.writeheader()
        writer.writerows(dictionaryData)

    return


def loadEmbeddings():
    model = loadModel()
    word_vectors = model.wv
    return word_vectors


# def convertLineToIndex

def convertLineToVector(line, word_vectors):
    token_size = 0

    # start of review sentence to be evaluated
    startingKey = "<SOS>"
    sos_vector = word_vectors.get_vector(startingKey, norm=False)
    result_vector = np.zeros((sos_vector.shape[0],), dtype=sos_vector.dtype)
    result_vector += sos_vector
    token_size += 1

    for token in line:
        if token in word_vectors.key_to_index:
            # for token in embedding
            known_vector = word_vectors.get_vector(token, norm=False)
            result_vector += known_vector
            token_size += 1
        else:
            # for unknown tokens
            print("token Not found")
            unknownKey = "<UNK>"
            unknown_vector = word_vectors.get_vector(unknownKey, norm=False)
            result_vector += unknown_vector
            token_size += 1

    # end of review sentence to be evaluated
    endingKey = "<SOS>"
    eos_vector = word_vectors.get_vector(endingKey, norm=False)
    result_vector += eos_vector
    token_size += 1

    result_vector /= token_size

    # print(indexData)
    return result_vector


def processLineToIndices(line, word_vectors):
    resultList = []

    # start of review sentence to be evaluated
    startingKey = "<SOS>"
    index = word_vectors.key_to_index[startingKey]

    resultList.append(index)

    for token in line:
        if token in word_vectors.key_to_index:
            # for token in embedding
            index = word_vectors.key_to_index[token]
            resultList.append(index)
        else:
            # for unknown tokens
            print("token Not found")
            unknownKey = "<UNK>"
            index = word_vectors.key_to_index[unknownKey]
            resultList.append(index)

    # end of review sentence to be evaluated
    endingKey = "<SOS>"
    index = word_vectors.key_to_index[endingKey]
    resultList.append(index)

    resultList = np.array(resultList)

    return resultList


def processTextToIndices(lines, word_vectors):
    result = []
    for line in lines:
        indices = processLineToIndices(line, word_vectors)
        result.append(indices)

    result = np.array(result, dtype='object')
    return result


def processData(folderPath, dataFileName):
    dataFilePath = os.path.join(folderPath, dataFileName)

    readingMode = 'r'
    tokens_array = []

    with open(dataFilePath, readingMode) as fileFP:
        reader = csv.reader(fileFP, delimiter=",")
        for i, line in enumerate(reader):
            np_line = np.array(line)
            tokens_array.append(np_line)

    tokens_array = np.array(tokens_array, dtype=object)

    return tokens_array


def processLabel(folderPath, dataFileName):
    labelFileName = "label_" + dataFileName
    labelFilePath = os.path.join(folderPath, labelFileName)

    readingMode = 'r'
    labels_array = []

    with open(labelFilePath, readingMode) as fileFP:
        reader = csv.reader(fileFP, delimiter=",")
        for i, line in enumerate(reader):
            labels_array.append(line[0])

    labels_array = np.array(labels_array)

    return labels_array
