import copy
import random
import sys, os, re, csv
from pathlib import Path


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


def writeInferenceOutput(labels, filename):

    print('dirname:     ', os.path.dirname(__file__))

    folderPath = os.path.dirname(__file__)
    if folderPath == "":
        folderPath = "."
    folderPath += "//data"

    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    labelFilePath = Path(folderPath + "//" + filename)

    print("writing to " + labelFilePath.__str__())
    with open(labelFilePath, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter="\n")
        writer.writerow(["labels"])
        writer.writerows([labels])

    print("writing to " + labelFilePath.__str__() + " complete")

    return

def extractLines(filePath):
    readingMode = 'r'
    lines = []

    with open(filePath, readingMode) as fileFP:
        for line in fileFP.readlines():
            lines.append(line)

    return lines