import os
from csv import DictReader, DictWriter
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

nltk.download('punkt')


def read(filepath):
    rows = []
    with open(filepath, "r", encoding='utf-8') as table:
        r = DictReader(table)

        for line in r:
            rows.append(line)
    return rows


def write(filepath, dictionary):
    with open(filepath, "w", encoding='utf-8', newline='') as table:
        fieldnames = list(dictionary[0].keys())
        print(fieldnames)
        writer = DictWriter(table, fieldnames=fieldnames)
        writer.writeheader()

        for line in dictionary:
            writer.writerow(line)

#Chunk generation
def generateChunks(essay, tokenizer):
    sentences = nltk.tokenize.sent_tokenize(essay)
    # initialize
    length = 0
    chunk = ""
    chunks = []
    count = -1
    for sentence in sentences:
        count += 1
        combined_length = len(
            tokenizer.tokenize(sentence)) + length  # add the no. of sentence tokens to the length counter
        if combined_length <= tokenizer.max_len_single_sentence:  # if it doesn't exceed
            chunk += sentence + " "  # add the sentence to the chunk
            length = combined_length  # update the length counter

            # if it is the last sentence
            if count == len(sentences) - 1:
                chunks.append(chunk.strip())  # save the chunk

        else:
            chunks.append(chunk.strip())  # save the chunk

            # reset
            length = 0
            chunk = ""

            # take care of the overflow sentence
            chunk += sentence + " "
            length = len(tokenizer.tokenize(sentence))

    return chunks


def create_summaries(path):
    checkpoint = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    for file in os.listdir(path):
        if file.endswith("_bodies.csv"):
            newFileName = str(file)
            newFileName = newFileName.replace("_bodies.csv", "_summaries.csv")
            filepath = os.path.join(path, file)
            articles = read(filepath)
            newArticles = []
            for article in articles:
                chunks = generateChunks(article['articleBody'], tokenizer)

                inputs = [tokenizer(chunk, return_tensors="pt") for chunk in chunks]
                summary = ""
                for input in inputs:
                    output = model.generate(**input)
                    summary = summary + (tokenizer.decode(*output, skip_special_tokens=True))

                print(summary)
                index = int(article['Body ID'])
                keyValue = {
                    "Body ID": index,
                    "articleSummary": summary
                }
                newArticles.append(keyValue)

            newFilePath = os.path.join(path, newFileName)
            write(newFilePath, newArticles)


if __name__ == "__main__":
    create_summaries("fnc-1")
