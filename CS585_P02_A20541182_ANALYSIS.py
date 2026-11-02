import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots to files without opening windows
import matplotlib.pyplot as plt

import CS585_P02A_A20541182 as part_a
import CS585_P02B_A20541182 as part_b

CORPUS_FILE_NAME = "source.txt"
TEST_WORDS = ["darcy", "elizabeth", "mother", "said", "london"]
NUMBER_OF_NEIGHBORS = 8
RANDOM_SEED = 0
PLOT_FOLDER = "analysis_plots"

# each D = 50 experiment changes one gradient descent setting from the baseline
BASELINE_SETTINGS = {"name": "baseline", "learning_rate": 0.05, "epochs": 5}
GRADIENT_EXPERIMENTS = [
    {"name": "lr_0.025", "learning_rate": 0.025, "epochs": 5},
    {"name": "lr_0.1", "learning_rate": 0.1, "epochs": 5},
    {"name": "epochs_15", "learning_rate": 0.05, "epochs": 15},
]


# Load the corpus once and build everything training needs
def prepare_training_data():
    tokens = part_a.load_and_tokenize(CORPUS_FILE_NAME)
    vocabulary, word_to_index = part_a.build_vocabulary(tokens)
    token_indices = []
    for token in tokens:
        token_indices.append(word_to_index[token])
    token_indices = np.array(token_indices)
    target_indices, context_indices = part_a.build_positive_pairs(token_indices, part_a.WINDOW_SIZE)
    sampling_distribution = part_a.build_negative_sampling_distribution(token_indices, len(vocabulary), part_a.SAMPLING_EXPONENT)
    return vocabulary, word_to_index, target_indices, context_indices, sampling_distribution


# Train one set of embeddings with the given settings and return (embeddings, training time)
def run_training(training_data, dimensions, settings):
    vocabulary, word_to_index, target_indices, context_indices, sampling_distribution = training_data

    # overwrite Part A's constants so train_embeddings uses this run's settings
    part_a.INITIAL_LEARNING_RATE = settings["learning_rate"]
    part_a.EPOCHS = settings["epochs"]
    part_a.SHOW_TRAINING_LOSS = True

    # same seed every run so differences come from the settings, not randomness
    np.random.seed(RANDOM_SEED)

    print(f"\n=== D = {dimensions}, {settings['name']} (lr {settings['learning_rate']}, {settings['epochs']} epochs) ===")
    start_time = time.time()
    embeddings = part_a.train_embeddings(target_indices, context_indices, sampling_distribution, len(vocabulary), dimensions)
    training_time = time.time() - start_time
    print(f"training time: {training_time:.2f} seconds")
    return embeddings, training_time


# Print each test word's neighbors, save its Part B plot, and return the neighbor lists
def report_neighbors_and_plot(embeddings, vocabulary, word_to_index, run_label):
    neighbors_by_word = {}
    for word in TEST_WORDS:
        word_index = word_to_index[word]
        neighbor_indices = part_b.find_nearest_neighbors(embeddings, word_index, NUMBER_OF_NEIGHBORS)
        neighbor_words = []
        for index in neighbor_indices:
            neighbor_words.append(vocabulary[index])
        neighbors_by_word[word] = neighbor_words
        print(f"  {word} -> {neighbor_words}")

        # Part B saves with a fixed name, so rename it to include this run's label
        selected_indices = [word_index] + neighbor_indices
        part_b.plot_vectors_3d(embeddings, vocabulary, selected_indices, word)
        dimensions = embeddings.shape[1]
        default_plot_name = f"CS585_P02B_A20541182_{word}_D{dimensions}.png"
        os.replace(default_plot_name, os.path.join(PLOT_FOLDER, f"{run_label}_{word}.png"))
        plt.close("all")
    return neighbors_by_word


# Count how many of each word's neighbors stayed the same compared to the baseline run
def print_neighbor_overlap(baseline_neighbors, experiment_neighbors):
    for word in TEST_WORDS:
        shared_count = 0
        for neighbor in experiment_neighbors[word]:
            if neighbor in baseline_neighbors[word]:
                shared_count += 1
        print(f"  {word}: {shared_count}/{NUMBER_OF_NEIGHBORS} neighbors same as baseline")


# Run every analysis experiment and print a results log
def main():
    os.makedirs(PLOT_FOLDER, exist_ok=True)
    training_data = prepare_training_data()
    vocabulary, word_to_index = training_data[0], training_data[1]

    print("##### PART 1: D = 50, 100, 200 with final settings #####")
    baseline_neighbors = None
    for dimensions in [50, 100, 200]:
        embeddings, training_time = run_training(training_data, dimensions, BASELINE_SETTINGS)
        neighbors = report_neighbors_and_plot(embeddings, vocabulary, word_to_index, f"D{dimensions}_baseline")
        if dimensions == 50:
            baseline_neighbors = neighbors

    print("\n##### PART 2: D = 50, changing gradient descent settings #####")
    for settings in GRADIENT_EXPERIMENTS:
        embeddings, training_time = run_training(training_data, 50, settings)
        neighbors = report_neighbors_and_plot(embeddings, vocabulary, word_to_index, f"D50_{settings['name']}")
        print_neighbor_overlap(baseline_neighbors, neighbors)


if __name__ == "__main__":
    main()