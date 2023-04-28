# NER using Bi-directional LSTM and Fine-tuning BERT

This repository contains notebooks for Named Entity Recognition (NER) using 
- A bi-directional LSTM neural network with glove pre-trained word embeddings in Tensorflow
- Fine-tuned BERT in PyTorch (using Huggingface)


## Dataset

The CoNLL-2003 dataset contains annotated data for three types of named entities: person names, organization names, and locations. The dataset is split into training, validation, and test sets.

The model is trained and evaluated on the corpus given by prof. Tanmoy Chakraborty. This dataset has two text files, train.txt and test.txt, with sentences labeled as "I-NP" or "O" where "I-NP" indicates the beginning of a noun phrase and "O" indicates no named entity.

## Libraries and Dependencies

The code uses the following python libraries and dependencies:

    Tensorflow
    Numpy
    Pandas
    Zipfile
    
    Pytorch
    Matplotlib

## Code Description

### The Bi-LSTM notebook includes the following steps:

- Data Extraction: The data is extracted from the zip file Assignment_2.zip and saved to /tmp directory.
- Text Processing: The text data is preprocessed by splitting the words and extracting labels for each sentence.
- Data Preparation: The data is prepared for training and testing by encoding the words and labels, and then padding the sequences to the same length.
- Word Embedding: GloVe pre-trained word embeddings are used for creating word vectors.
- Model Architecture: The model architecture includes an embedding layer, bi-directional LSTM layer, and a dense layer with sigmoid activation for classification.
- Model Compilation: The model is compiled using binary cross-entropy loss and Adam optimizer.
- Model Training: The model is trained on the prepared data with 10 epochs.
- Model Evaluation: The trained model is evaluated on the test data with accuracy as the evaluation metric.

### The BERT notebook includes the following steps:
- Load the dataset: The dataset used in this code is available in the /kaggle/input/ner-data/ner.csv file. The dataset contains two columns, text and labels.
- Tokenize the dataset: The BertTokenizerFast class is used to tokenize the dataset. The my_tokenizer function is used to tokenize the text column of the dataset.
- Encode the labels: The unique_labels dictionary is used to encode the labels.
- Split the dataset: The dataset is split into four parts, df_hold, df_train, df_val, and df_test.
- Tokenize the columns: The tokenize_column function is used to preprocess the columns of the dataset and return input ids, attention masks, and token type ids as 2D arrays.
- Categorical encoding: The cat_labels function is used to perform categorical encoding on the labels.
- Model training: The BertForTokenClassification class is used to train the model overnight for 2 epochs
 
## Author

This notebook was written by Bhavneek Singh.
