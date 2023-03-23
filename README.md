# NER using Bi-directional LSTM

This is a python code for text classification using a bi-directional LSTM (Long Short-Term Memory) neural network with glove pre-trained word embeddings.

## Dataset

The CoNLL-2003 dataset contains annotated data for three types of named entities: person names, organization names, and locations. The dataset is split into training, validation, and test sets.

The model is trained and evaluated on the corpus given by prof. Tanmoy Chakraborty. This dataset has two text files, train.txt and test.txt, with sentences labeled as "I-NP" or "O" where "I-NP" indicates the beginning of a noun phrase and "O" indicates no named entity.

## Libraries and Dependencies

The code uses the following python libraries and dependencies:

    Tensorflow
    Numpy
    Pandas
    Zipfile

## Code Description

### The python code includes the following steps:

- Data Extraction: The data is extracted from the zip file Assignment_2.zip and saved to /tmp directory.
- Text Processing: The text data is preprocessed by splitting the words and extracting labels for each sentence.
- Data Preparation: The data is prepared for training and testing by encoding the words and labels, and then padding the sequences to the same length.
- Word Embedding: GloVe pre-trained word embeddings are used for creating word vectors.
- Model Architecture: The model architecture includes an embedding layer, bi-directional LSTM layer, and a dense layer with sigmoid activation for classification.
- Model Compilation: The model is compiled using binary cross-entropy loss and Adam optimizer.
- Model Training: The model is trained on the prepared data with 10 epochs.
- Model Evaluation: The trained model is evaluated on the test data with accuracy as the evaluation metric.

## Usage

    Clone the repository on your local machine.
    Extract the Assignment_2.zip file to the root directory of the cloned repository.
    Open the Jupyter Notebook file text_classification_bi_lstm.ipynb using Jupyter Notebook or Google Colab.
    Run each cell of the notebook to execute the code.
 
## Author

This notebook was written by Bhavneek Singh.
