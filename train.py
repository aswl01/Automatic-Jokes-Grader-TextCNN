import argparse
import os
from datetime import datetime
import tensorflow as tf
from text_cnn import Text_CNN
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
# import string
import codecs
import time
from sklearn.cross_validation import train_test_split
import numpy as np
from tensorflow.contrib import learn


def main(args):
    if args.pretrained_model:
        pretrained_model = tf.train.latest_checkpoint(args.pretrained_model)
        print('Pre-trained model: %s' % os.path.expanduser(pretrained_model))

    corpus_path = 'jokes/'
    true_label_path = 'popular_jokes.txt'
    false_label_path = 'non_popular_jokes.txt'
    jokes = []
    max_sentence_length = 0
    labels = []
    for counter, label_path in enumerate([false_label_path, true_label_path]):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for file_name in lines:
                file_name = file_name.strip()
                file_name += ".txt"
                path = os.path.join(corpus_path, file_name)
                with codecs.open(path, encoding='utf-8') as text_file:
                    texts = text_file.readlines()
                    joke = ""
                    for text in texts:
                        # list of tokens in one text file after preprocess
                        joke += " " + preprocess(text)
                    if joke == "":
                        continue
                    joke = joke[1:]
                    jokes.append(joke)
                    labels.append([1 - counter, counter])
                    max_sentence_length = max(max_sentence_length, len(joke.split(" ")))

    print("finish preprocessing")
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
    jokes = np.array(list(vocab_processor.fit_transform(jokes)))
    print("sentence length = {}, voc_size = {}".format(max_sentence_length, len(vocab_processor.vocabulary_)))

    x_train, x_dev, y_train, y_dev = train_test_split(jokes, labels, test_size=0.1, random_state=42)

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=gpu_options
            )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = Text_CNN(sentence_length=max_sentence_length, num_classes=2, vocab_size=len(vocab_processor.vocabulary_),
                           embedding_size=args.embedding_dim,
                           filter_sizes=list(map(int, args.filter_sizes.split(","))), num_filters=args.num_filters,
                           l2_reg_lambda=args.l2_reg_lambda)
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.num_checkpoints)
            sess.run(tf.global_variables_initializer())

            if args.pretrained_model:
                pretrained_model = tf.train.latest_checkpoint(args.pretrained_model)
                print('Restoring the latest pretrained model: %s' % pretrained_model)
                saver.restore(sess, pretrained_model)

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: args.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            batches = batch_iter(
                list(zip(x_train, y_train)), args.batch_size, args.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % args.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % args.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


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


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # Data loading params
    parser.add_argument('--dev_sample_percentage', type=float,
                        help='Percentage of the training data to use for validation', default=0.1)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)

    # Model Hyperparameters
    parser.add_argument('--embedding_dim', type=int,
                        help='Dimensionality of character embedding (default: 128)', default=128)
    parser.add_argument('--filter_sizes', type=str,
                        help="Comma-separated filter sizes (default: 2,3,4,5,6')", default="2, 3, 4, 5, 6")
    parser.add_argument('--num_filters', type=int,
                        help='Number of filters per filter size (default: 128)', default=128)
    parser.add_argument('--dropout_keep_prob', type=float,
                        help='Dropout keep probability (default: 0.5)', default=0.5)
    parser.add_argument('--l2_reg_lambda', type=float,
                        help='L2 regularization lambda (default: 0.0)', default=0.0)

    # Training parameters
    parser.add_argument('--batch_size', type=int,
                        help='Batch Size (default: 64)', default=64)
    parser.add_argument('--num_epochs', type=int,
                        help='Number of training epochs (default: 200)', default=200)
    parser.add_argument('--evaluate_every', type=int,
                        help='Evaluate model on dev set after this many steps (default: 100)', default=100)
    parser.add_argument('--checkpoint_every', type=int,
                        help='Save model after this many steps (default: 100)', default=100)
    parser.add_argument('--num_checkpoints', type=int,
                        help='Number of checkpoints to store (default: 3)', default=3)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
