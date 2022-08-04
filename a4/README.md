# Report: Assigmnent 4

### Name: Arefin Shimon, Shaikh Shawon

### Student ID: 20942174

### Course: MSCI641 (Spring 2022)

### output of inference.py is generated in the data folder when the inference code is run. Also visible in command line.
### Note: Submission delayed by a day. Discussed with instructor in class for late submission. followed tutorial and adapted parts of tutorial code - discussed with instructor in class regarding tutorial code adoption. Model trained in CUDA. 

## Table
| Hidden Layer Name | Dropout Rate 	| Accuracy (test set)   |
| ----------------- | ------------- | --------------------- |
|relu	            |0.1	        |0.761575               |
|relu	            |0.2	        |0.7639                 |
|relu	            |0.3	        |0.767225               |
|relu	            |0.5	        |0.7725375              |
|relu	            |0.7	        |0.7707875              |
|relu	            |0.9	        |0.7485375              |
|sigmoid	        |0.1	        |0.768875               |
|sigmoid	        |0.2	        |0.772525               |
|sigmoid	        |0.3	        |0.7728875              |
|sigmoid	        |0.5	        |0.7728375              |
|sigmoid	        |0.7	        |0.7678375              |
|sigmoid	        |0.9	        |0.749675               |
|tanh	            |0.1	        |0.758375               |
|tanh           	|0.2	        |0.7616875              |
|tanh	            |0.3	        |0.7608375              |
|tanh	            |0.5	        |0.7590125              |
|tanh	            |0.7	        |0.748675               |
|tanh	            |0.9	        |0.7309                 |
     


## Explaination (upto 10 sentences)
The model was trained on data with stopwords based on our findings from A2. In A2, we discovered that the classification accuracy was lower on dataset without stopwords, as they were also removing contextual positive and negative words. So we trained our dataset on regular data with stopwords. The above table shows different activation function results with L2 norm regularization with different dropout rates. For all the activation functions, the accuracy increases with an increase with in dropout rate and then starts falling beyond a threshold. Dropout turns off some connections at the output layer to reduce overfitting, but beyond a certain threshold, removing connections start underfitting the model - hence the result observations. Sigmoid at 0.3 dropout with L2 activation has the best observed results with ReLU following closely behind. Relu has better results than tanh because of insensitivity to vanishing gradiant problem. Adding the L2 regularization improved results. This is because L2 regularization dispersed error terms in all the weights and able to learn complex patterns to tune out the weights. 