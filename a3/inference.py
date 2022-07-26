import gensim.models
import nltk as nltk
import sys, os, re
from pathlib import Path
from nltk.corpus import stopwords
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

def loadModel():
    folderPath = os.path.dirname(__file__) + "//data"
    modelfilePath = folderPath + "//w2v.model"
    print(modelfilePath)
    model = None
    if os.path.exists(modelfilePath):
        model = gensim.models.Word2Vec.load(modelfilePath)
        print("Found Existing Model")
    else:
        print("Model does not exist, run main.py to train the model first")

    return model

class WordLoader(object):
    def __init__(self, datapath):
        self.evalFilePath = datapath

    def __iter__(self):
        readingMode = 'r'

        with open(self.evalFilePath, readingMode, newline='\n') as evalFP:
            for line in evalFP.readlines():
                line_strip = line.strip()
                yield line_strip

if __name__ == '__main__':
    inference_file_path = sys.argv[1]  # Get the filepath
    model = loadModel()
    wordIter = WordLoader(inference_file_path)
    topn = 20

    base_file_name = os.path.basename(inference_file_path)
    output_file_name = "output_"+base_file_name

    folderPath = os.path.dirname(__file__) + "//data"
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    modelfilePath = folderPath + "//" + output_file_name

    print("writing to " + modelfilePath.__str__())
    with open(modelfilePath, 'w') as outputFile:

        for word in wordIter:
            similarWordList = []
            if word in model.wv:
                print(word + " is present in the vocabulary of model")
                similarWordList = model.wv.most_similar(positive=[word], topn=topn)
            else:
                print(word + " is not present in the vocabulary of model")
            print(similarWordList)
            outputFile.write(similarWordList.__str__())
            outputFile.write('\n')