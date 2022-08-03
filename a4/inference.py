import copy
import random
import sys, os, re, csv
from pathlib import Path
from tokenizer import *
import numpy as np
import torch
from models import FC_FF_NN_Model
from helper import *


def tokens2indicesForInference(tokens, word_vectors, max_sent_length, device):
    sos_id = word_vectors.key_to_index["<SOS>"]
    eos_id = word_vectors.key_to_index["<EOS>"]
    pad_id = word_vectors.key_to_index["<PAD>"]
    unk_id = word_vectors.key_to_index["<UNK>"]

    resultList = [pad_id] * max_sent_length
    resultList[0] = (sos_id)

    i = 1
    for token in tokens:
        if token in word_vectors.key_to_index:
            # for token in embedding
            index = word_vectors.key_to_index[token]
            resultList[i] = index
        else:
            # for unknown tokens
            print("token Not found")
            resultList[i] = unk_id

        i += 1
        if (i >= max_sent_length):
            break

    resultList[len(resultList) - 1] = eos_id
    return resultList

if __name__ == '__main__':
    inputFilePath = sys.argv[1]  # Get the inputFilePath
    classifierName = sys.argv[2]  # Get the classifierName
    print(inputFilePath)
    print(classifierName)

    modelFolderPath = os.path.dirname(__file__)
    if modelFolderPath == "":
        modelFolderPath = "."
    modelFolderPath += "//data"
    print(modelFolderPath)
    modelFileName = "nn_" + classifierName + ".model"
    modelFilePath = Path(modelFolderPath + "//" + modelFileName)
    print(modelFilePath)
    if os.path.exists(modelFilePath):
        print("Model Exists")
        if (os.path.exists(inputFilePath)):
            print("Input file exists")

            if not os.path.exists(modelFolderPath):
                os.makedirs(modelFolderPath)

            inputFileName = os.path.basename(inputFilePath.rsplit(".", 1)[0])
            outputFileName = "outputA4_" + inputFileName + ".csv"
            classes = ["False", "True"]

            loadedModel = torch.load(modelFilePath)
            device = next(loadedModel.parameters()).device
            loadedModel = loadedModel.to(device)
            print(loadedModel)


            lines = extractLines(inputFilePath)
            labels = []
            indiceArray = []
            for line in lines:
                tokens = tokenizeLine(line)
                indices = tokens2indicesForInference(tokens, loadEmbeddings(), 30, device)
                indiceArray.append(indices)
            indiceArray = torch.tensor(indiceArray).to(device)
            results = loadedModel(indiceArray)
            for result in results:
                print(result)
                print(result.argmax(-1))
                output = classes[result.argmax(-1)]
                labels.append(output)
            for i,line in enumerate(lines):
                print(line + " | Result : " + labels[i])
            writeInferenceOutput(labels,outputFileName)

        else:
            print("Input file does not exists")
    else:
        print("Input model does not exist")



