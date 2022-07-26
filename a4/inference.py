import copy
import random
import sys, os, re, csv
from pathlib import Path
from tokenizer import *
import numpy as np
import torch
from main import FC_FF_NN_Model

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
            print(loadedModel)

            lines = extractLines(inputFilePath)
            labels = []
            for line in lines:
                tokens = tokenizeLine(line)
                print(tokens)
                output = classes[loadedModel(tokens).argmax(1)]
                labels.append(output)
                print(str(tokens) + " : " + output)
            print(labels)
            writeInferenceOutput(labels,outputFileName)

        else:
            print("Input file does not exists")
    else:
        print("Input model does not exist")



