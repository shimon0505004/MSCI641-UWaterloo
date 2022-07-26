import csv
import time
import random

import gensim.models
import nltk as nltk
import sys, os, re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from helper import *
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, textDatas, labels):
        'Initialization'
        self.labels = labels
        self.text_Datas = textDatas

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.text_Datas)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.text_Datas[index]
        y = self.labels[index]

        return X, y


class FC_FF_NN_Model(nn.Module):
    def __init__(self, word_vectors, hiddenLayer, dropoutRate):
        super(FC_FF_NN_Model, self).__init__()
        self.learningRate = 5
        self.word_vectors = word_vectors
        self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(word_vectors.vectors))
        self.hidden = hiddenLayer  # nn.Sigmoid()
        embedding_dim = word_vectors.vector_size
        num_class = 2
        self.fc = nn.Linear(embedding_dim, num_class)
        self.finalLayer = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=dropoutRate)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learningRate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)


    def forward(self, inputLine):
        hidden_layer_input = self.hiddenLayerInput(inputLine)
        translationLayer_layer_input = self.hidden(hidden_layer_input)
        dropout_layer_input = self.fc(translationLayer_layer_input)
        final_layer_input = self.dropout(dropout_layer_input)
        final_layer_output = self.finalLayer(final_layer_input)
        print(final_layer_output)
        return final_layer_output

    def hiddenLayerInput(self, line):
        indicesList = processLineToIndices(line, self.word_vectors)
        sum_of_embeddings = None
        number_of_embeddings = 0
        for indices in indicesList:
            embeddingTensor = self.embeddings(torch.LongTensor([indices]))
            if sum_of_embeddings is None:
                sum_of_embeddings = embeddingTensor
            else:
                sum_of_embeddings = embeddingTensor + sum_of_embeddings

            number_of_embeddings += 1
        average_for_input_layer = torch.div(sum_of_embeddings, number_of_embeddings)
        return average_for_input_layer

    def train_dataset(self, trainData, trainLabel):
        # stochastic gradient descent, adding L2-norm regularization

        model.train()
        # for epoch in range(epochSize):
        total_acc, total_count = 0, 0
        trainingAccuracies = []
        accuracy = 0.

        randomIndexList = list(range(len(trainData)))
        random.shuffle(randomIndexList)

        for i in randomIndexList:
            data = trainData[i]
            self.optimizer.zero_grad()
            y_real = None
            label = trainLabel[i]
            if label == "True":
                y_real = torch.LongTensor([1])
            else:
                y_real = torch.LongTensor([0])

            classes = ["False","True"]

            output_probablity = model(data)
            print("Probability")
            print(output_probablity)
            print(y_real)
            loss = self.loss(output_probablity, y_real)
            # compute gradients
            loss.backward()
            # update model parameters
            self.optimizer.step()
            total_acc += (output_probablity.argmax(1) == y_real).sum().item()
            total_count += 1
            accuracy = total_acc / total_count
            print("Training Line " + str(i) + " Accuracy: " + str(accuracy) + " Real Label: " + label + " Perceived Label: " + classes[output_probablity.argmax(1)])
        return accuracy

    def evaluate_dataset(self, evalData, evalLabel):

        model.eval()
        # for epoch in range(epochSize):
        total_acc, total_count = 0, 0
        accuracy = 0.

        randomIndexList = list(range(len(evalData)))
        random.shuffle(randomIndexList)

        with torch.no_grad():
            for i in randomIndexList:
                data = evalData[i]
                y_real = None
                label = evalLabel[i]
                if label == "True":
                    y_real = torch.LongTensor([1])
                else:
                    y_real = torch.LongTensor([0])

                output_probablity = model(data)
                loss = self.loss(output_probablity, y_real)
                total_acc += (output_probablity.argmax(1) == y_real).sum().item()
                total_count += 1
                accuracy = total_acc / total_count

        return accuracy

    def train_and_validate_dataset(self, trainData, trainLabel, validData, validLabel, EPOCHS):

        total_accu = None
        training_accuracy = 0.
        validation_accuracy = 0.

        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()
            training_accuracy = self.train_dataset(trainData, trainLabel)
            validation_accuracy = self.evaluate_dataset(validData, validLabel)
            if total_accu is not None and total_accu > validation_accuracy:
                self.scheduler.step()
            else:
                total_accu = validation_accuracy
            print('-' * 70)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                  'valid accuracy {:8.3f} '.format(epoch,
                                                   time.time() - epoch_start_time,
                                                   validation_accuracy))
            print('-' * 70)

        return training_accuracy, validation_accuracy


if __name__ == '__main__':
    folderPath = sys.argv[1]  # Get the filepath
    # folderPath = os.path.dirname(__file__) + "//data"

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    trainDataPath = "train.csv"
    validDataPath = "val.csv"
    testDataPath = "test.csv"

    trainData = processData(folderPath, trainDataPath)
    validData = processData(folderPath, validDataPath)
    testData = processData(folderPath, testDataPath)

    trainLabel = processLabel(folderPath, trainDataPath)
    validLabel = processLabel(folderPath, validDataPath)
    testLabel = processLabel(folderPath, testDataPath)

    classes = ("False", "True")

    word_vectors = loadEmbeddings()

    hiddenLayers = [{
        "name": "relu",
        "model": nn.ReLU(),
        "file": "nn_relu.model",
    },
        {
            "name": "sigmoid",
            "model": nn.Sigmoid(),
            "file": "nn_sigmoid.model",
        },
        {
            "name": "tanh",
            "model": nn.Tanh(),
            "file": "nn_tanh.model",
        },
    ]
    dropoutRates = [0.2]

    dataPath = os.path.dirname(__file__)
    if dataPath == "":
        dataPath = "."
    dataPath = dataPath + "//data//"
    outputFileName = "output.csv"
    csv_header = ["Hidden Layer Name", "Dropout Rate", "Accuracy (test set)","Filepath"]
    csv_data = []

    for hiddenLayer in hiddenLayers:
        best_accuracy = None
        for dropoutRate in dropoutRates:
            model = FC_FF_NN_Model(word_vectors, hiddenLayer["model"], dropoutRate=dropoutRate)
            training_accuracy, validation_accuracy = model.train_and_validate_dataset(trainData,
                                                                                      trainLabel,
                                                                                      validData,
                                                                                      validLabel,
                                                                                      EPOCHS=2)
            testing_accuracy = model.evaluate_dataset(testData, testLabel)
            if best_accuracy is not None and best_accuracy > testing_accuracy:
                # Dont save model
                print("Model not saved")
            else:
                best_accuracy = testing_accuracy
                modelFilePath = dataPath + hiddenLayer["file"]
                modelFilePath = Path(modelFilePath).__str__()
                torch.save(model, modelFilePath)
            # saving all model file regardless
            modelFilePath2 = dataPath + "nn_" + str(dropoutRate).replace('.', '_') + "_" + hiddenLayer[
                "name"] + ".model"
            modelFilePath2 = Path(modelFilePath2).__str__()
            torch.save(model, modelFilePath2)

            data = {
                csv_header[0]: hiddenLayer["name"],
                csv_header[1]: dropoutRate,
                csv_header[2]: testing_accuracy,
                csv_header[3]: modelFilePath2,
            }
            csv_data.append(data)

    writeReportToCSV(csv_data,outputFileName)