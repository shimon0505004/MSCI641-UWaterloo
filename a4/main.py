import sys
from helper import *
from dataloader import make_dataloader
from models import FC_FF_NN_Model

import torch
from tqdm import tqdm


def train(dloader, model, criterion, optimizer):
    model.train()
    losses, acc = [], []
    total_acc = 0
    print(len(dloader))
    for batch in tqdm(dloader):
        y = batch["label"]
        #print(batch["length"])
        logits = model(batch["input_ids"])
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = torch.argmax(logits, -1)
        total_acc += torch.sum((preds==y))
        acc.append((preds == y).float().mean().item())

    print(
        f"Train Loss: {np.array(losses).mean():.4f} | Train Accuracy: {np.array(acc).mean():.4f}"
    )

    return np.array(acc).mean()


@torch.no_grad()
def test(dloader, model, criterion):
    model.eval()
    losses, acc = [], []
    for batch in dloader:
        y = batch["label"]
        logits =model(batch["input_ids"])
        loss = criterion(logits, y)
        losses.append(loss.item())
        preds = torch.argmax(logits, -1)
        acc.append((preds == y).float().mean().item())

    print(f"Loss: {np.array(losses).mean():.4f} | Accuracy: {np.array(acc).mean():.4f}")

    return np.array(acc).mean()

def main():
    folderPath = sys.argv[1]  # Get the filepath
    # folderPath = os.path.dirname(__file__) + "//data"

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    word_vectors = loadEmbeddings()

    trainDataPath = "train.csv"
    validDataPath = "val.csv"
    testDataPath = "test.csv"

    trainData = processData(folderPath, trainDataPath)
    validData = processData(folderPath, validDataPath)
    testData = processData(folderPath, testDataPath)

    trainLabel = processLabel(folderPath, trainDataPath)
    validLabel = processLabel(folderPath, validDataPath)
    testLabel = processLabel(folderPath, testDataPath)

    train_dloader = make_dataloader(trainData, trainLabel, word_vectors, max_sent_length=30, batch_size=128,
                                    device=device)
    valid_dloader = make_dataloader(validData, validLabel, word_vectors, max_sent_length=30, batch_size=128,
                                    device=device)
    test_dloader = make_dataloader(testData, testLabel, word_vectors, max_sent_length=30, batch_size=128,
                                   device=device)

    hiddenLayers = [{
        "name": "relu",
        "model": torch.nn.ReLU(),
        "file": "nn_relu.model",
    },
        {
            "name": "sigmoid",
            "model": torch.nn.Sigmoid(),
            "file": "nn_sigmoid.model",
        },
        {
            "name": "tanh",
            "model": torch.nn.Tanh(),
            "file": "nn_tanh.model",
        },
    ]

    dropoutRates = [0.1,0.2, 0.3, 0.5, 0.7,0.9]
    index = 0

    dataPath = os.path.dirname(__file__)
    if dataPath == "":
        dataPath = "."
    dataPath = dataPath + "//data//"
    outputFileName = "output.csv"
    csv_header = ["Hidden Layer Name", "Dropout Rate", "Accuracy (test set)", "Filepath"]
    csv_data = []

    for hiddenLayer in hiddenLayers:
        best_accuracy = None
        for dropoutRate in dropoutRates:
            model = FC_FF_NN_Model(word_vectors, hiddenLayer=hiddenLayer["model"], dropoutRate=dropoutRate, n_classes=2)
            model.to(device)
            print(model)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)

            for epoch in range(20):
                print(f"===Epoch {epoch}===")
                training_accuracy = train(train_dloader, model, criterion, optimizer)
                print("Validating...")
                validation_accuracy = test(valid_dloader, model, criterion)
                print("Testing...")
                testing_accuracy = test(test_dloader, model, criterion)

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

    writeReportToCSV(csv_data, outputFileName)


if __name__ == '__main__':
    main()
