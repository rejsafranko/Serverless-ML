import re

import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.models import Sequential

from argparse import ArgumentParser, Namespace


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    return parser.parse_args()


def remove_missing_data(df: pd.DataFrame):
    """
    Removes missing data and empty strings from a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing sentences.

    Returns:
    - pd.DataFrame: DataFrame with missing data and empty strings removed.

    Note:
    - This function drops rows with missing values and sentences that are empty strings from the input DataFrame.
    - After removing the rows, it resets the index of the DataFrame.
    """

    df = df.dropna()
    df = df[df["sentence"] != ""]
    df = df.reset_index(drop=True)
    return df


def remove_stopwords(sentence: str):
    """
    Removes stopwords from a given sentence.

    Parameters:
    - sentence (str): The input sentence.

    Returns:
    - str: The input sentence with stopwords removed.

    Note:
    - This function utilizes the NLTK library to tokenize the input sentence into words and then
      removes stopwords from it using the English stopwords list provided by NLTK.
    - It returns the input sentence with stopwords removed.
    """

    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words]
    return " ".join(filtered_sentence)


def clean_sentence(sentence: str):
    """
    Extracts the last word from each sentence in a DataFrame and creates a new column.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the sentences.

    Returns:
    - df (pd.DataFrame): DataFrame with an additional column containing the last word of each sentence.

    Note:
    - This function adds a new column named "last_word" to the input DataFrame, which contains
      the last word of each sentence.
    - It also removes the last word from each sentence and updates the "sentence" column accordingly.
    """

    # Remove non-alphabetical characters and leave single whitespaces.
    cleaned_sentence = re.sub(r"[^a-zA-Z\s]", "", sentence)

    # Replace multiple whitespaces with a single whitespace.
    cleaned_sentence = re.sub(r"\s+", " ", cleaned_sentence)

    return (
        cleaned_sentence.lower().strip()
    )  # Strip leading and trailing whitespaces and lowercase.


def extract_last_word(df: pd.DataFrame):
    # Extract last word from each sentence and place it in a new column.
    df["last_word"] = df["sentence"].apply(lambda x: x.split(" ")[-1])
    # Remove the last word from each sentence.
    df["sentence"] = df["sentence"].apply(lambda x: " ".join(x.split()[:-1]))
    return df


def preprocess_data(data_path: str):
    """
    Preprocesses textual data for sequence modeling tasks.

    Parameters:
    - data_path (str): The path to the data file.

    Returns:
    - dataset (dict): A dictionary containing preprocessed data and related information.
      It has the following keys:
      - "features" (numpy.ndarray): Input features, padded sequences of tokenized sentences.
      - "labels" (numpy.ndarray): Output labels, representing the last word in each sentence.
    - vocab_size (int): The size of the vocabulary after tokenization.
    - max_len (int): The maximum length of padded sequences.

    Note:
    - This function reads text data from the specified file path, removes links, tokenizes
      the text into sentences, removes stopwords, and cleans sentences by removing non-alphabetic
      characters, extra whitespaces, and converting text to lowercase.
    - It tokenizes the cleaned sentences, pads the sequences to a uniform length, and extracts
      the last word from each sentence as the label.
    - The function utilizes the Tokenizer class from the Keras preprocessing module for tokenization.
    - If a token in the label sequence is not found in the vocabulary, it is replaced with the out-of-vocabulary
      token "<UNK>".
    - The preprocessed data is returned along with the vocabulary size and the maximum sequence length.
    """

    with open(data_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Remove links from the text.
    text = re.sub(r"http\S+", "", text)

    # Tokenize text into sentences.
    sentences = sent_tokenize(text)
    df = pd.DataFrame(sentences, columns=["sentence"])

    # Remove any leading or trailing whitespaces.
    df["sentence"] = df["sentence"].str.strip()

    # Remove missing values and empty strings.
    df = remove_missing_data(df)

    # Remove stopwords.
    df["sentence"] = df["sentence"].apply(remove_stopwords)

    # Clean sentences - remove non-alphabetic characters, leave only single whitespaces and normalize the text (lowercase conversion).
    df["sentence"] = df["sentence"].apply(clean_sentence)

    # Create label colum called "last_word".
    df = extract_last_word(df)

    # Tokenization.
    tokenizer = Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(df["sentence"])
    vocab_size = len(tokenizer.word_index) + 1

    # Convert sentences to sequences.
    sequences = tokenizer.texts_to_sequences(df["sentence"])

    # Pad sequences.
    max_len = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post")

    # Create input and output sequences.
    X = np.array(padded_sequences)
    y_text = df["last_word"]

    # Tokenize output data.
    y_sequences = tokenizer.texts_to_sequences(y_text)

    # If token not in vocabulary, replace it with oov token <UNK>.
    for i, seq in enumerate(y_sequences):
        if len(seq) == 0:
            y_sequences[i] = [tokenizer.word_index["<UNK>"]]

    # Convert labels to numpy array.
    y = np.array([sequence[0] for sequence in y_sequences if len(sequence) != 0])

    dataset = dict()
    dataset["features"] = X
    dataset["labels"] = y

    return dataset, vocab_size, max_len


def build_model(vocab_size: int, max_len: int, embedding_dim=50):
    """
    Builds a sequential neural network model for natural language processing tasks.

    Parameters:
    - vocab_size (int): The size of the vocabulary, i.e., the number of unique words in the corpus.
    - max_len (int): The maximum length of input sequences.
    - embedding_dim (int, optional): The dimensionality of the embedding space. Defaults to 50.

    Returns:
    - model: A compiled Keras sequential model for NLP tasks.

    Note:
    - This function constructs a neural network with an embedding layer, bidirectional LSTM layer,
      and a dense softmax output layer. It uses sparse categorical crossentropy as the loss function,
      Adam optimizer, and accuracy as the evaluation metric.
    """

    model = Sequential()
    model.add(
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)
    )
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(units=vocab_size, activation="softmax"))
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model


def main(args: Namespace):
    dataset, vocab_size, max_len = preprocess_data(args.data_path)
    model = build_model(vocab_size, max_len)

    # Train the model.
    num_epochs = 30
    model.fit(dataset["features"], dataset["labels"], epochs=num_epochs, verbose=1)

    # Save the model.
    model.save(args.save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
