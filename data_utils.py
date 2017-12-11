from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np


def preprocess(query):
    # create English stop words list
    # stop_words = set(stopwords.words('english'))
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # ascii = set(string.printable)
    # lower cace tokens
    tokenizer = RegexpTokenizer('\s+', gaps=True)
    query = query.lower()
    tokens = tokenizer.tokenize(query)
    # only keep lowercace letter, remove numbers,punctuation, non_english word
    filtered_tokens = []
    for token in tokens:
        if all(ord(c) < 128 for c in token):
            # token = str(token).translate(None, string.punctuation)
            #            token = token.translate(None, string.digits)
            filtered_tokens.append(token)

    # remove stop words from tokens
    # tokens = [i for i in filtered_tokens if i not in stop_words]

    # # Create wordNetLemmatizer only extract NN and NNS to improve accuracy
    # lem = WordNetLemmatizer()
    # # extract only noun
    # tagged_text = nltk.pos_tag(tokens)
    # for word, tag in tagged_text:
    #     words.append({"word": word, "pos": tag})
    # tokens = [lem.lemmatize(word["word"]) for word in words if word["pos"] in ["NN", "NNS"]]

    # stem tokens
    tokens = [p_stemmer.stem(token) for token in tokens]
    tokens = " ".join(tokens)
    return tokens


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
