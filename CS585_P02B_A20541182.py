import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

STUDENT_HEADER = "Konan, Otioh, A20541182 solution:"
EMBEDDING_FILE_NAME = "CS585_P02A_A20541182_EMBED.TXT"
DEFAULT_NEIGHBORS = 5
REDUCTION_METHOD = "PCA"
SHOW_NEIGHBOR_LIST = False


# Validate command line arguments and return (word, number of neighbors)
def parse_arguments():
    # sys.argv[0] is the script name, so 2 user arguments means length 3
    if len(sys.argv) != 3:
        print("ERROR: expected 2 arguments: python CS585_P02B_A20541182.py WORD K")
        sys.exit(1)

    word_argument, neighbors_argument = sys.argv[1], sys.argv[2]

    # exit if the embedding file from Part A isn't in the current folder
    if not os.path.isfile(EMBEDDING_FILE_NAME):
        print(f"ERROR: embedding file '{EMBEDDING_FILE_NAME}' not found")
        sys.exit(1)

    # WORD must be non-empty and made only of Latin letters
    if re.fullmatch(r"[A-Za-z]+", word_argument) is None:
        print(f"ERROR: '{word_argument}' must be a non-empty word with only Latin letters")
        sys.exit(1)

    # anything outside 1-10 (or not an integer) falls back to the default
    try:
        number_of_neighbors = int(neighbors_argument)
        if number_of_neighbors < 1 or number_of_neighbors > 10:
            number_of_neighbors = DEFAULT_NEIGHBORS
    except ValueError:
        number_of_neighbors = DEFAULT_NEIGHBORS

    return word_argument.lower(), number_of_neighbors


# Load the Part A CSV into a vocabulary list, a word -> row index lookup, and an embedding matrix
def load_embeddings(embedding_file_name):
    vocabulary = []
    word_to_index = {}
    embedding_rows = []
    with open(embedding_file_name, "r") as embedding_file:
        for line in embedding_file:
            # split the line into the word and its embedding values, convert the values to floats, and store them
            parts = line.strip().split(", ")
            word = parts[0]
            values = []
            for value in parts[1:]:
                values.append(float(value))
            word_to_index[word] = len(vocabulary)
            vocabulary.append(word)
            embedding_rows.append(values)
    return vocabulary, word_to_index, np.array(embedding_rows)


# Return the row indices of the K words with the highest cosine similarity to the given word (excluding itself)
def find_nearest_neighbors(embeddings, word_index, number_of_neighbors):
    # divide every vector by its length so a dot product equals cosine similarity
    vector_lengths = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / vector_lengths
    similarities = normalized_embeddings @ normalized_embeddings[word_index]

    # sort from most to least similar and skip the word itself
    sorted_indices = np.argsort(-similarities)
    neighbor_indices = []
    for index in sorted_indices:
        if index != word_index:
            neighbor_indices.append(index)
        if len(neighbor_indices) == number_of_neighbors:
            break
    return neighbor_indices


# Fit PCA on the whole vocabulary, project the chosen words to 3D, draw them as arrows from the origin, and save a PNG
def plot_vectors_3d(embeddings, vocabulary, selected_indices, word):
    # fitting on every word gives the same 3 axes for any WORD and works even when K is 1 or 2
    pca = PCA(n_components=3)
    pca.fit(embeddings)
    points_3d = pca.transform(embeddings[selected_indices])

    figure = plt.figure(figsize=(9, 7))
    axes = figure.add_subplot(projection="3d")

    for point_number in range(len(selected_indices)):
        x_value, y_value, z_value = points_3d[point_number]
        label = vocabulary[selected_indices[point_number]]
        # the query word is red, its neighbors are blue
        if point_number == 0:
            color = "red"
        else:
            color = "blue"
        axes.quiver(0, 0, 0, x_value, y_value, z_value, color=color, arrow_length_ratio=0.08)
        axes.text(x_value, y_value, z_value, label, color=color, fontsize=11)

    # make sure every arrow tip and the origin are inside the view
    largest_value = np.abs(points_3d).max() * 1.2
    axes.set_xlim(-largest_value, largest_value)
    axes.set_ylim(-largest_value, largest_value)
    axes.set_zlim(-largest_value, largest_value)

    dimensions = embeddings.shape[1]
    axes.set_title(f"'{word}' and its {len(selected_indices) - 1} nearest neighbors (D = {dimensions}, PCA to 3D)")
    axes.set_xlabel("PC1")
    axes.set_ylabel("PC2")
    axes.set_zlabel("PC3")

    plot_file_name = f"CS585_P02B_A20541182_{word}_D{dimensions}.png"
    figure.savefig(plot_file_name, dpi=150, bbox_inches="tight")


# Run Part B: load embeddings, find WORD's nearest neighbors, plot them in 3D
def main():
    word, number_of_neighbors = parse_arguments()
    vocabulary, word_to_index, embeddings = load_embeddings(EMBEDDING_FILE_NAME)

    if word not in word_to_index:
        print(f"ERROR: '{word}' is not in the vocabulary")
        sys.exit(1)

    word_index = word_to_index[word]
    neighbor_indices = find_nearest_neighbors(embeddings, word_index, number_of_neighbors)

    print(STUDENT_HEADER)
    print(f"Word: {word}")
    print(f"Number of neighbors: {number_of_neighbors}")
    print(f"Dimensionality reduction method: {REDUCTION_METHOD}")

    # off by default so on-screen output matches the spec; turn on for the report
    if SHOW_NEIGHBOR_LIST:
        neighbor_words = []
        for index in neighbor_indices:
            neighbor_words.append(vocabulary[index])
        print(f"Neighbors: {neighbor_words}")

    selected_indices = [word_index] + neighbor_indices
    plot_vectors_3d(embeddings, vocabulary, selected_indices, word)
    plt.show()


if __name__ == "__main__":
    main()