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

def processData(folderPath, dataFileName):
    dataFilePath = os.path.join(folderPath, dataFileName)

    readingMode = 'r'
    tokens_array = []

    with open(dataFilePath, readingMode) as fileFP:
        reader = csv.reader(fileFP, delimiter=",")
        for i, line in enumerate(reader):
            #np_line = np.array(line)
            tokens_array.append(line)

    #tokens_array = np.array(tokens_array, dtype=object)

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

    #labels_array = np.array(labels_array)

    return labels_array
