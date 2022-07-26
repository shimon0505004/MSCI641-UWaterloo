import copy
import random
import sys, os, re, csv
from pathlib import Path

stopword_list = [
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "ain", "ain't", "aint",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are", "are't", "aren", "aren't", "arent"
                                      "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can", "can't", "cant",
    "could", "couldn", "couldn't", "couldnt",
    "d",
    "did", "didn", "didn't", "didnt",
    "do", "does", "doesn", "doesn't", "doing", "don", "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had", "hadn", "hadn't", "hadnt", "has", "hasn", "hasn't", "hasnt",
    "how",
    "have", "haven", "haven't", "havent", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "i",
    "ing"
    "if",
    "in", "into",
    "is", "isn", "isn't", "isnt",
    "it", "it's", "its", "itself",
    "just",
    "ll",
    "m", "ma", "me",
    "might", "mightn", "mightn't", "mightnt",
    "more",
    "most",
    "must", "mustn", "mustn't", "mustnt",
    "my", "myself",
    "need", "needn", "needn't", "neednt",
    "no", "nor", "not",
    "o", "of", "off",
    "on", "once", "only", "or", "other",
    "our", "ours", "ourself", "ourselves",
    "out",
    "over", "own",
    "re", "s", "same",
    "shan", "shan't", "shant",
    "she", "she's", "should", "should've", "shouldve", "shouldn", "shouldn't", "shouldnt",
    "so", "some",
    "such",
    "t", "than", "that", "that'll", "thatll",
    "the", "their", "theirs", "them", "themselves", "then",
    "there",
    "these",
    "they", "they're", "they've", "theyre", "theyve",
    "this",
    "those",
    "through",
    "to", "too",
    "under", "until",
    "up", "upto",
    "ve",
    "very",
    "was", "wasn", "wasn't", "wasnt",
    "we", "we're", "we've", "weve",
    "were", "weren", "weren't", "werent",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who", "whom",
    "why",
    "will",
    "with",
    "won", "won't", "wont",
    "would", "wouldn", "wouldn't", "wouldnt",
    "y",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
]

stopword_list = [stopword.lower() for stopword in stopword_list]
stopword_set = set(stopword_list)


# method tokenizes a line without stopwords
def tokenizeLine(line):
    # preprocessing to lower case
    modifiedLine1 = line.lower()

    modifiedLine2 = re.sub('[\t\n]', ' ', modifiedLine1)

    listOfCharactersInModifiedLine2 = list(modifiedLine2)
    # need to deal with special cases of '.' removal
    index = 0
    for pos, char in enumerate(modifiedLine2):
        prevPos = pos - 1
        nextPos = pos + 1

        if (char == '.'):
            if (prevPos >= 0) and (not modifiedLine2[prevPos].isnumeric()):
                listOfCharactersInModifiedLine2[index] = ' '

            if (nextPos < len(modifiedLine2)) and (not modifiedLine2[nextPos].isnumeric()):
                listOfCharactersInModifiedLine2[index] = ' '

        if char == ('`') or char == ('\''):
            if ((prevPos >= 0) and (nextPos < len(modifiedLine2))):
                if (modifiedLine2[nextPos].isalpha() and modifiedLine2[prevPos].isalpha()):
                    listOfCharactersInModifiedLine2[index] = '\''
                else:
                    listOfCharactersInModifiedLine2[index] = ' '
            else:
                listOfCharactersInModifiedLine2[index] = ' '

        if char.isalpha():
            if (nextPos < len(modifiedLine2)) and modifiedLine2[nextPos].isnumeric():
                listOfCharactersInModifiedLine2.insert(index + 1, ' ')
                index += 1

            if (prevPos >= 0) and modifiedLine2[prevPos].isnumeric():
                listOfCharactersInModifiedLine2.insert(index, ' ')
                index += 1

        index += 1

    modifiedLine2 = "".join(listOfCharactersInModifiedLine2)
    modifiedLine2 = " " + modifiedLine2 + " "  # appending whitespace for easier parsing. will be trimmed later.
    seperatorPattern = r"[ '_~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^]"

    # analyzing possible apostophie rules
    modifiedLine2 = re.sub(r"(?<=[ _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(i'v|i've|iv'e|ive')"
                           r"(?=[a-zA-z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " i have ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[ _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"('ve|v'e|ve')"
                           r"(?=[a-zA-z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " have ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(didn't|didnt'|did'nt|did't|didnt)"
                           r"(?=[a-zA-z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " did not ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(isn't|is'nt)"
                           r"(?=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " is not ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(i'd)"
                           r"(?=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " i would ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z])"
                           r"('r')"
                           r"(?=[a-zA-Z])",
                           " are ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z])"
                           r"('n')"
                           r"(?=[a-zA-Z])",
                           " and ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"([ _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(y'all)"
                           r"([ _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " you all ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(i'm)"
                           r"(?=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " i am ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z])"
                           r"('ing)"
                           r"(?=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " ing ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[ _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(it's)"
                           r"(?=[ _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " it is ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z])"
                           r"(n't|'nt)"
                           r"(?=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " not ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[d])"
                           r"('t|t')"
                           r"(?=[ _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " not ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z])"
                           r"('re|r'e)"
                           r"(?=[ _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " are ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z])"
                           r"(r'e)"
                           r"(?=[ _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " are ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z])"
                           r"('v|'ve|v'e|ve')"
                           r"(?=[ _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " have ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z])"
                           r"(l'l|'ll|ll')"
                           r"(?=[ _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " will ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z])"
                           r"('d|d')"
                           r"(?=[ _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " would ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z])"
                           r"('s|s')"
                           r"(?=[ _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[ _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(c'mon|cm'on|cmo'n|cmon)"
                           r"(?=[ _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])",
                           " come on ",
                           modifiedLine2)

    modifiedLine2 = re.sub(r"(?<=[0-9])"
                           r"(\"|&#34;)",
                           " inch ",
                           modifiedLine2  # processing 9"
                           )

    modifiedLine2 = re.sub(r"(?<=[0-9])"
                           r"(\')",
                           " feet ",
                           modifiedLine2  # processing 9'
                           )

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(&Agrave;|&Aacute;|&Acirc;|&Atilde;|&Auml;|&Aring;|&agrave;|&aacute;|&acirc;|&atilde;|&auml;|&aring;)",
                           "A",
                           modifiedLine2
                           )  # special character code removal

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(&AElig;|&aelig;)",
                           "ae",
                           modifiedLine2
                           )  # special character code removal

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(&AElig;|&aelig;)",
                           "ae",
                           modifiedLine2
                           )  # special character code removal

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(&szlig;)",
                           "b",
                           modifiedLine2
                           )  # special character code removal

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"((&Ccedil;)|(&ccedil;))",
                           "c",
                           modifiedLine2
                           )  # special character code removal

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(&Egrave|&Eacute;|&Ecirc;|&Euml;|&egrave;|&eacute;|&ecirc;|&euml;)",
                           "e",
                           modifiedLine2
                           )  # special character code removal

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(&#131)",
                           "f",
                           modifiedLine2
                           )  # special character code removal

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(&#131;|&Igrave;|&Iacute;|&Icirc;|&Iuml;)",
                           "l",
                           modifiedLine2
                           )  # special character code removal

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(&igrave;|&iacute;|&icirc;|&iuml;)",
                           "i",
                           modifiedLine2
                           )  # special character code removal

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(&Ntilde;|&ntilde;)",
                           "n",
                           modifiedLine2
                           )  # special character code removal

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(&Ograve;|&Oacute;|&Ocirc;|&Otilde;|&Ouml;|&ograve;|&oacute;|&ocirc;|&otilde;|&ouml;|&Oslash;|&oslash;)",
                           "o",
                           modifiedLine2
                           )  # special character code removal

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(&#140;|&#156;)",
                           "oe",
                           modifiedLine2
                           )  # special character code removal

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"((&#138;)|(&#154;))",
                           "s",
                           modifiedLine2
                           )  # special character code removal

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(&Ugrave;)",
                           "u",
                           modifiedLine2
                           )  # special character code removal

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"((&Uacute;)|(&Ucirc;)|(&Uuml;)|(&ugrave;)|(&uacute;)|(&ucirc;)|(&uuml;))",
                           "u",
                           modifiedLine2
                           )  # special character code removal

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(&Yacute;|&yacute;|&#159;|&yuml;)",
                           "y",
                           modifiedLine2
                           )  # special character code removal

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"(&215;)",
                           "*",
                           modifiedLine2
                           )  # special character code removal

    modifiedLine2 = re.sub(r"(?<=[a-zA-Z _~\-|@!?:;()/\\{}%&*<=>\[\],\t\n\"#$+/^])"
                           r"((&[a-z]+;)|(&#[0-9]+;))",
                           " ",
                           modifiedLine2
                           )  # Removing quotes

    modifiedLine2 = re.sub(r"^\s+|\s+$", "", modifiedLine2)  # removing leading and trailing spaces in line

    # removalPattern = '[\"#$+\/\\^]'
    words = re.split(seperatorPattern, str(modifiedLine2))

    wordsWithoutPattern = []
    for word in words:
        modifiedWord = word
        if len(modifiedWord) > 0:
            wordsWithoutPattern.extend([modifiedWord])

    return wordsWithoutPattern


def writeLinesWithLabelsToCSVFile(filename, linesWithLabel):
    print('dirname:     ', os.path.dirname(__file__))
    folderPath = os.path.dirname(__file__) + "//data"
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    datafilePath = Path(folderPath + "//" + filename)
    labelFilePath = Path(folderPath + "//label_" + filename)

    lines = []
    labels = []
    for row in linesWithLabel:
        lines.append(row["tokens"])
        labels.append([str(row["label"])])

    print("writing to " + datafilePath.__str__())
    with open(datafilePath, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=",")
        writer.writerows(lines)

    print("writing to " + datafilePath.__str__() + " complete")

    print("writing to " + labelFilePath.__str__())
    with open(labelFilePath, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=",")
        writer.writerows(labels)

    print("writing to " + labelFilePath.__str__() + " complete")

    return


def extractLines(dataFolderPath, filename):
    filePath = os.path.join(dataFolderPath, filename);
    readingMode = 'r'
    lines = []

    with open(filePath, readingMode) as fileFP:
        for line in fileFP.readlines():
            lines.append(line)

    return lines



def getUnfilteredDataWithLabels(dictList):
    outputList = []
    for item in dictList:
        output = {
            "tokens": item["tokens"],
            "label": item["label"],
        }
        outputList.append(output)

    return outputList

def getFilteredDataWithLabels(dictList):
    outputList = []
    for item in dictList:
        output = {
            "tokens": item["filteredTokens"],
            "label": item["label"],
        }
        outputList.append(output)

    return outputList

def buildDictListWithLabelsFromLines(lines, fileName, folderPath, label, filterSet):
    dictionaryList = []

    index = 0
    for line in lines:

        allTokens = []
        filteredTokens = []
        for token in tokenizeLine(line):
            allTokens.extend([token])
            if token.lower() not in filterSet:
                filteredTokens.extend([token])

        dictionary = {
            "tokens": allTokens,
            "filteredTokens": filteredTokens,
            "label": label,
        }
        dictionaryList.append(dictionary)
        print("Token from line " + str(index + 1) + " in file "
              + fileName + " at Path: " + folderPath + " completed. "
                                                       "Label: " + str(label))
        index = index + 1

    print("Token Parsing completed in file " + fileName + " at Path: " + folderPath)

    copiedList = copy.deepcopy(dictionaryList)
    random.shuffle(copiedList)

    print("Original list size: " + str(len(copiedList)))

    validationListSize = (int)(len(copiedList) // 10)
    testingListSize = validationListSize

    validationListStartIndex = 0
    validationListEndIndex = validationListSize
    testingListStartIndex = validationListEndIndex
    testingListEndIndex = testingListStartIndex + testingListSize
    trainingListStartIndex = testingListEndIndex
    trainingListEndIndex = len(copiedList)

    validationList = copiedList[validationListStartIndex:validationListEndIndex]
    testingList = copiedList[testingListStartIndex:testingListEndIndex]
    trainingList = copiedList[trainingListStartIndex:trainingListEndIndex]

    print("---------------------------------------------------------------")
    print("For File : " + fileName)
    print("training list size: " + str(len(trainingList)) + " Percentage: " + str(
        len(trainingList) / len(dictionaryList) * 100) + "%")
    print("validation list size: " + str(len(validationList)) + " Percentage: " + str(
        len(validationList) / len(dictionaryList) * 100) + "%")
    print("testing list size: " + str(len(testingList)) + " Percentage: " + str(
        len(testingList) / len(dictionaryList) * 100) + "%")
    print("---------------------------------------------------------------")

    ws_list = {
        "out": getUnfilteredDataWithLabels(dictionaryList),
        "train": getUnfilteredDataWithLabels(trainingList),
        "val": getUnfilteredDataWithLabels(validationList),
        "test": getUnfilteredDataWithLabels(testingList),
    }

    wo_list = {
        "out": getFilteredDataWithLabels(dictionaryList),
        "train": getFilteredDataWithLabels(trainingList),
        "val": getFilteredDataWithLabels(validationList),
        "test": getFilteredDataWithLabels(testingList),
    }

    output = {
        "with_stopwords": ws_list,
        "without_stopwords": wo_list,
    }

    return output


def buildDictListWithLabels(fileName, folderPath, label, filterSet):
    lines = extractLines(folderPath, fileName)

    print("Generating Dataset including and excluding Stopwords")
    output = buildDictListWithLabelsFromLines(lines, fileName, folderPath, label, filterSet)
    print("Generating Dataset including and excluding Stopwords completed")

    return output


def mergeDataset(postive, negative):
    original = []
    training = []
    validation = []
    test = []

    original.extend(postive["out"])
    original.extend(negative["out"])
    training.extend(postive["train"])
    training.extend(negative["train"])
    validation.extend(postive["val"])
    validation.extend(negative["val"])
    test.extend(postive["test"])
    test.extend(negative["test"])
    dictionary = {
        "out": original,
        "train": training,
        "val": validation,
        "test": test,
    }
    return dictionary


def writeToOutputFile(dictionaries, suffix):
    for fileNamePrefix, dictionary in dictionaries.items():
        filename = str(fileNamePrefix) + str(suffix) + ".csv"
        writeLinesWithLabelsToCSVFile(filename,
                                      dictionary)

    return


def generateDataset(folderPath,
                    positiveFileName,
                    negativeFileName,
                    stopwords):
    positiveData = buildDictListWithLabels(positiveFileName, folderPath, True, stopwords)
    negativeData = buildDictListWithLabels(negativeFileName, folderPath, False, stopwords)

    allDataWithStopwords = mergeDataset(positiveData["with_stopwords"],
                                        negativeData["with_stopwords"])

    allDataWithoutStopwords = mergeDataset(positiveData["without_stopwords"],
                                           negativeData["without_stopwords"])

    allDataWithStopwordsSuffix = ""
    writeToOutputFile(allDataWithStopwords, allDataWithStopwordsSuffix)

    allDataWithoutStopwordsSuffix = "_ns"
    writeToOutputFile(allDataWithoutStopwords, allDataWithoutStopwordsSuffix)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    folderPath = sys.argv[1]  # Get the filepath
    positiveFileName = "pos.txt"
    negativeFileName = "neg.txt"

    # print(sorted(stopword_set))
    generateDataset(folderPath=folderPath,
                    positiveFileName=positiveFileName,
                    negativeFileName=negativeFileName,
                    stopwords=stopword_set)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
