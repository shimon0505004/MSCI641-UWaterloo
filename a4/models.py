import torch


class FC_FF_NN_Model(torch.nn.Module):
    def __init__(self, word_vectors, hiddenLayer, dropoutRate, n_classes):
        super(FC_FF_NN_Model, self).__init__()
        self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.FloatTensor(word_vectors.vectors))
        embedding_dim = word_vectors.vector_size
        hidden_dim = int(3*embedding_dim)
        self.fc1 = torch.nn.Linear(30*embedding_dim, hidden_dim)
        self.hidden = hiddenLayer
        self.dropout = torch.nn.Dropout(p=dropoutRate)
        self.fc2 = torch.nn.Linear(hidden_dim, n_classes)
        self.finalLayer = torch.nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding_layer(x)
        embedded = torch.flatten(embedded, start_dim=1)
        fc1_output = self.fc1(embedded)
        hidden_output = self.hidden(fc1_output)
        dropout_output = self.dropout(hidden_output)
        fc2_output = self.fc2(dropout_output)
        softmax_output = self.finalLayer(fc2_output)
        return softmax_output