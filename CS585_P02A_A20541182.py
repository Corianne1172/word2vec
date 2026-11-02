import os
import re
import sys
import time

import numpy as np

STUDENT_HEADER = "Konan, Otioh, A20541182 solution:"
OUTPUT_FILE_NAME = "CS585_P02A_A20541182_EMBED.TXT"
DEFAULT_DIMENSIONS = 50
WINDOW_SIZE = 2
NEGATIVES_PER_POSITIVE = 2
SAMPLING_EXPONENT = 0.75
INITIAL_LEARNING_RATE = 0.05
EPOCHS = 5
BATCH_SIZE = 512
SHOW_TRAINING_LOSS = False

# Validate command line arguments and return (dimensions, corpus file name)
def parse_arguments():
    # sys.argv[0] is the script name, so 2 user arguments means length 3
    if len(sys.argv) != 3:
        print("ERROR: expected 2 arguments: python CS585_P02A_A20541182.py D CORPUS_FILE")
        sys.exit(1)

    dimensions_argument, corpus_file_name = sys.argv[1], sys.argv[2]

    # exit if the corpus file isn't in the current folder
    if not os.path.isfile(corpus_file_name):
        print(f"ERROR: corpus file '{corpus_file_name}' not found")
        sys.exit(1)

    # anything that isn't a positive integer falls back to the default
    try:
        dimensions = int(dimensions_argument)
        if dimensions <= 0:
            dimensions = DEFAULT_DIMENSIONS
    except ValueError:
        dimensions = DEFAULT_DIMENSIONS

    return dimensions, corpus_file_name


# Read the corpus, keep only Latin letters, lowercase, and split into tokens
def load_and_tokenize(corpus_file_name):
    with open(corpus_file_name, "r", encoding="utf-8", errors="ignore") as corpus_file:
        raw_text = corpus_file.read()
    # replace every run of non-letters (punctuation, digits, newlines) with a single space
    latin_only_text = re.sub(r"[^A-Za-z]+", " ", raw_text).lower()
    return latin_only_text.split()


# Build the sorted vocabulary and a word -> row index lookup for the embedding matrices
def build_vocabulary(tokens):
    vocabulary = sorted(set(tokens))
    word_to_index = {}
    for index, word in enumerate(vocabulary):
        word_to_index[word] = index
    return vocabulary, word_to_index

# Build every (target, context) positive pair within the sliding window, as two parallel index arrays
def build_positive_pairs(token_indices, window_size):
    target_list = []
    context_list = []
    for position in range(len(token_indices)):
        # look at every neighbor from window_size words back to window_size words ahead
        for offset in range(-window_size, window_size + 1):
            neighbor_position = position + offset
            # skip the target itself and neighbors that fall off either end of the text
            if offset == 0 or neighbor_position < 0 or neighbor_position >= len(token_indices):
                continue
            target_list.append(token_indices[position])
            context_list.append(token_indices[neighbor_position])
    return np.array(target_list), np.array(context_list)


# Build the probability of drawing each vocabulary word as a negative sample
def build_negative_sampling_distribution(token_indices, vocabulary_size, exponent):
    # word_counts[i] = how many times vocabulary word i appears in the corpus
    word_counts = np.bincount(token_indices, minlength=vocabulary_size)
    weighted_counts = word_counts ** exponent
    return weighted_counts / weighted_counts.sum()


# Draw k random negative context words for every positive pair
def sample_negatives(sampling_distribution, number_of_pairs, negatives_per_positive):
    vocabulary_size = len(sampling_distribution)
    return np.random.choice(vocabulary_size, size=(number_of_pairs, negatives_per_positive), p=sampling_distribution)

# Squash dot products into probabilities between 0 and 1
def sigmoid(values):
    # clip so np.exp doesn't overflow on very large scores
    clipped_values = np.clip(values, -10, 10)
    return 1 / (1 + np.exp(-clipped_values))


# Train target and context matrices with skip-gram negative sampling and return the target matrix as the embeddings
def train_embeddings(target_indices, context_indices, sampling_distribution, vocabulary_size, dimensions):
    # both matrices start as small random values, one row per vocabulary word
    target_matrix = (np.random.rand(vocabulary_size, dimensions) - 0.5) / dimensions
    context_matrix = (np.random.rand(vocabulary_size, dimensions) - 0.5) / dimensions

    number_of_pairs = len(target_indices)
    total_pairs_to_process = EPOCHS * number_of_pairs
    pairs_processed = 0

    for epoch in range(EPOCHS):
        # new pair order and new negatives every epoch
        shuffled_order = np.random.permutation(number_of_pairs)
        negative_indices = sample_negatives(sampling_distribution, number_of_pairs, NEGATIVES_PER_POSITIVE)
        epoch_loss = 0.0

        for batch_start in range(0, number_of_pairs, BATCH_SIZE):
            batch = shuffled_order[batch_start:batch_start + BATCH_SIZE]
            batch_targets = target_indices[batch]
            batch_contexts = context_indices[batch]
            batch_negatives = negative_indices[batch]

            # learning rate shrinks linearly toward 0 over all of training
            learning_rate = INITIAL_LEARNING_RATE * max(1 - pairs_processed / total_pairs_to_process, 0.0001)

            target_vectors = target_matrix[batch_targets]
            positive_vectors = context_matrix[batch_contexts]

            # σ(t · c_pos) for every pair in the batch
            positive_scores = sigmoid(np.sum(target_vectors * positive_vectors, axis=1))
            # (σ(t · c_pos) - 1) as a column so it scales each row of vectors
            positive_error = (positive_scores - 1).reshape(-1, 1)

            # slide 129 gradients: positive part
            target_gradient = positive_error * positive_vectors
            positive_gradient = positive_error * target_vectors
            epoch_loss -= np.sum(np.log(positive_scores))

            # slide 129 gradients: one negative at a time
            negative_gradients = []
            for negative_number in range(NEGATIVES_PER_POSITIVE):
                negative_vectors = context_matrix[batch_negatives[:, negative_number]]
                negative_scores = sigmoid(np.sum(target_vectors * negative_vectors, axis=1))
                negative_error = negative_scores.reshape(-1, 1)
                target_gradient += negative_error * negative_vectors
                negative_gradients.append(negative_error * target_vectors)
                epoch_loss -= np.sum(np.log(1 - negative_scores))

            # np.add.at handles the same word showing up multiple times in one batch
            np.add.at(context_matrix, batch_contexts, -learning_rate * positive_gradient)
            for negative_number in range(NEGATIVES_PER_POSITIVE):
                np.add.at(context_matrix, batch_negatives[:, negative_number], -learning_rate * negative_gradients[negative_number])
            np.add.at(target_matrix, batch_targets, -learning_rate * target_gradient)

            pairs_processed += len(batch)

        # off by default so on-screen output matches the spec; turn on for experiments
        if SHOW_TRAINING_LOSS:
            print(f"epoch {epoch + 1}/{EPOCHS}, average loss: {epoch_loss / number_of_pairs:.4f}")

    return target_matrix

# Write one line per vocabulary word in alphabetical order: word, value1, value2, ...
def save_embeddings(output_file_name, vocabulary, embeddings):
    with open(output_file_name, "w") as output_file:
        for index in range(len(vocabulary)):
            formatted_values = []
            for value in embeddings[index]:
                formatted_values.append(f"{value:.6f}")
            output_file.write(vocabulary[index] + ", " + ", ".join(formatted_values) + "\n")

# Run the full Part A pipeline: load corpus, train word2vec, save embeddings, print the summary
def main():
    dimensions, corpus_file_name = parse_arguments()
    tokens = load_and_tokenize(corpus_file_name)
    vocabulary, word_to_index = build_vocabulary(tokens)

    # convert the token list to row indices once so training works with numbers, not strings
    token_indices = []
    for token in tokens:
        token_indices.append(word_to_index[token])
    token_indices = np.array(token_indices)

    target_indices, context_indices = build_positive_pairs(token_indices, WINDOW_SIZE)
    sampling_distribution = build_negative_sampling_distribution(token_indices, len(vocabulary), SAMPLING_EXPONENT)

    start_time = time.time()
    embeddings = train_embeddings(target_indices, context_indices, sampling_distribution, len(vocabulary), dimensions)
    training_time = time.time() - start_time

    save_embeddings(OUTPUT_FILE_NAME, vocabulary, embeddings)

    print(STUDENT_HEADER)
    print(f"Number of dimensions: {dimensions}")
    print(f"Training file name: {corpus_file_name}")
    print(f"Embedding file name: {OUTPUT_FILE_NAME}")
    print(f"Training time: {training_time:.2f} seconds")

if __name__ == "__main__":
    main()