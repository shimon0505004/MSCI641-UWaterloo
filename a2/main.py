"""Driver script to train an MNB classifier for sentiment classifiction"""
import csv
import os
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import metrics
from pathlib import Path
import pickle


def doNothing(doc):
    return doc


def processData(folderPath, dataFileName):
    dataFilePath = os.path.join(folderPath, dataFileName)

    readingMode = 'r'
    tokens_array = []

    with open(dataFilePath, readingMode) as fileFP:
        reader = csv.reader(fileFP, delimiter=",")
        for i, line in enumerate(reader):
            tokens_array.append(line)

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

    # print(labels_array)

    return labels_array


def generateModel(dataFolderPath, trainingData, validationData, testingData, lowerBoundNGram, upperBoundNgram):
    X_train = processData(dataFolderPath, trainingData)
    Y_train = processLabel(dataFolderPath, trainingData)

    nGramVectorizer = CountVectorizer(
        tokenizer=doNothing,
        preprocessor=doNothing,
        ngram_range=(lowerBoundNGram, upperBoundNgram)
    )


    # summarize encoded vector
    X_train_vector = nGramVectorizer.fit_transform(X_train)
    X_valid = processData(dataFolderPath, validationData)
    Y_valid = processLabel(dataFolderPath, validationData)
    X_valid_vector = nGramVectorizer.transform(X_valid)

    """
    mnb = MultinomialNB()

    # fit the model on training data
    mnb.fit(X_train_vector, Y_train)
    # define the parameter values that should be searched for MNB
    alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0, 1.0, 5.0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
             80, 85, 90, 95, 100]
    # create a parameter grid: map the parameter names to the values that should be searched for MNB
    param_grid = dict(alpha=alpha)
    gridsearch_mnb = GridSearchCV(mnb, param_grid, cv=10, scoring='accuracy')
    """
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0, 1.0, 5.0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
             80, 85, 90, 95, 100]

    accuracies = []
    for alpha in alphas:
        mnb1 = MultinomialNB(alpha=alpha)
        # fit the model on training data
        mnb1.fit(X_train_vector, Y_train)
        accuracy = mnb1.score(X_valid_vector, Y_valid)
        accuracies.append(accuracy)

    max_accuracy = max(accuracies)
    max_index = accuracies.index(max_accuracy)
    optimal_alpha = alphas[max_index]

    for i in range(0, len(accuracies)):
        print(str(alphas[i]) + " , " + str(accuracies[i]))

    # tuning hyperparameters by using validation data
    mnb = MultinomialNB(alpha=optimal_alpha)
    mnb.fit(X_train_vector, Y_train)


    X_test = processData(dataFolderPath, testingData)
    Y_test = processLabel(dataFolderPath, testingData)
    X_test_vector = nGramVectorizer.transform(X_test)

    accuracy = mnb.score(X_test_vector, Y_test)
    print(accuracy)

    result = {
        "model": mnb,
        "accuracy": accuracy,
        "vectorizer": nGramVectorizer,
    }

    return result


def generateModelsAndData(dataFolderPath,
                          trainingData,
                          validationData,
                          testingData,
                          lowerBoundNGram,
                          upperBoundNgram,
                          stopwordsRemoved,
                          textFeatures,
                          fileName,
                          ):
    modelResult = generateModel(dataFolderPath, trainingData, validationData, testingData, lowerBoundNGram,
                                upperBoundNgram)
    result = {
        "model": modelResult["model"],
        "accuracy": modelResult["accuracy"],
        "vectorizer": modelResult["vectorizer"],
        "stopwordsRemoved": stopwordsRemoved,
        "textFeatures": textFeatures,
        "pickleFileName": fileName,
    }
    return result


def writeModelToPickleFile(mnb, filename):
    print('dirname:     ', os.path.dirname(__file__))
    folderPath = os.path.dirname(__file__) + "//data"
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    datafilePath = Path(folderPath + "//" + filename)

    with open(datafilePath, 'wb') as f:
        pickle.dump(mnb, f)

    return

def writeVectorizerToPickleFile(vectorizer, filename):
    print('dirname:     ', os.path.dirname(__file__))
    folderPath = os.path.dirname(__file__) + "//data"
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    datafilePath = Path(folderPath + "//vectorizer_" + filename)

    with open(datafilePath, 'wb') as f:
        pickle.dump(vectorizer, f)

    return

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


if __name__ == '__main__':
    dataFolderPath = sys.argv[1]  # Get the filepath
    training_ws_data = "train.csv"
    validation_ws_data = "val.csv"
    testing_ws_data = "test.csv"
    training_ns_data = "train_ns.csv"
    validation_ns_data = "val_ns.csv"
    testing_ns_data = "test_ns.csv"

    unigram_no_stopwords_removed = generateModelsAndData(
        dataFolderPath=dataFolderPath,
        trainingData=training_ws_data,
        validationData=validation_ws_data,
        testingData=testing_ws_data,
        lowerBoundNGram=1,
        upperBoundNgram=1,
        stopwordsRemoved="no",
        textFeatures="unigrams",
        fileName="mnb_uni.pkl",
    )

    bigram_no_stopwords_removed = generateModelsAndData(
        dataFolderPath=dataFolderPath,
        trainingData=training_ws_data,
        validationData=validation_ws_data,
        testingData=testing_ws_data,
        lowerBoundNGram=2,
        upperBoundNgram=2,
        stopwordsRemoved="no",
        textFeatures="bigrams",
        fileName="mnb_bi.pkl",
    )

    uni_and_bigram_no_stopwords_removed = generateModelsAndData(
        dataFolderPath=dataFolderPath,
        trainingData=training_ws_data,
        validationData=validation_ws_data,
        testingData=testing_ws_data,
        lowerBoundNGram=1,
        upperBoundNgram=2,
        stopwordsRemoved="no",
        textFeatures="unigrams+bigrams",
        fileName="mnb_uni_bi.pkl",
    )

    unigram_stopwords_removed = generateModelsAndData(
        dataFolderPath=dataFolderPath,
        trainingData=training_ns_data,
        validationData=validation_ns_data,
        testingData=testing_ns_data,
        lowerBoundNGram=1,
        upperBoundNgram=1,
        stopwordsRemoved="yes",
        textFeatures="unigrams",
        fileName="mnb_uni_ns.pkl",
    )

    bigram_stopwords_removed = generateModelsAndData(
        dataFolderPath=dataFolderPath,
        trainingData=training_ns_data,
        validationData=validation_ns_data,
        testingData=testing_ns_data,
        lowerBoundNGram=2,
        upperBoundNgram=2,
        stopwordsRemoved="yes",
        textFeatures="bigrams",
        fileName="mnb_bi_ns.pkl",
    )

    uni_and_bigram_stopwords_removed = generateModelsAndData(
        dataFolderPath=dataFolderPath,
        trainingData=training_ns_data,
        validationData=validation_ns_data,
        testingData=testing_ns_data,
        lowerBoundNGram=1,
        upperBoundNgram=2,
        stopwordsRemoved="yes",
        textFeatures="unigrams+bigrams",
        fileName="mnb_uni_bi_ns.pkl",
    )

    results = [unigram_stopwords_removed,
               bigram_stopwords_removed,
               uni_and_bigram_stopwords_removed,
               unigram_no_stopwords_removed,
               bigram_no_stopwords_removed,
               uni_and_bigram_no_stopwords_removed,
               ]

    csv_header = ["Stopwords removed", "text features", "Accuracy (test set)"]
    csv_data = []

    for model_details in results:
        writeModelToPickleFile(model_details["model"],
                               model_details["pickleFileName"])

        writeVectorizerToPickleFile(model_details["vectorizer"],
                               model_details["pickleFileName"])

        data = {
            csv_header[0]: model_details["stopwordsRemoved"],
            csv_header[1]: model_details["textFeatures"],
            csv_header[2]: model_details["accuracy"],
        }
        csv_data.append(data)

    writeReportToCSV(csv_data, "results.csv")
