# Import the packages
import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from Levenshtein import distance as get_distance
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
import tiktoken

# Hide future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set the parameters
input_folder = f"../data/responses"
output_folder = f"../data/results"

def get_jaccard(a, b):

    # Tokenize the strings into sets of words
    set1 = set(a)
    set2 = set(b)

    # Calculate the Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    similarity = intersection / union if union != 0 else 0.0

    return similarity

def get_jaccard_simiarity(texts):

    # Get all combinations of texts as tuples
    combos = list(combinations(texts, 2))

    # Calculate the Jaccard similarity for each tuple
    similarities = [get_jaccard(a, b) for a, b in combos]

    # Return the mean similarity
    return np.mean(similarities)

def get_bag_of_words_similarity(texts):

    # Create the vectorizer
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(texts)

    # Get the cosine similarity
    similarity_matrix = cosine_similarity(vectors)

    # Get the upper triangle
    upper_matrix = np.triu_indices_from(similarity_matrix, k=1)

    # Get the average similarity
    avg_similarity = np.mean(similarity_matrix[upper_matrix])

    return avg_similarity

def get_tfidf_similarity(texts):

    # Create the vectorizer
    vectorizer = TfidfVectorizer()

    # Create the vectors
    vectors = vectorizer.fit_transform(texts)

    # Calculate the cosine similarity
    similarity_matrix = cosine_similarity(vectors)

    upper_matrix = np.triu_indices_from(similarity_matrix, k=1)

    avg_similarity = np.mean(similarity_matrix[upper_matrix])

    return avg_similarity

def get_levenshtein_distance(texts):

    distances = [get_distance(a, b) for a, b in combinations(texts, 2)]

    avg_distance = np.mean(distances)

    return avg_distance

def get_bleu_score(texts):

    # Calculate the average BLEU score
    bleu_scores = [sentence_bleu([a], b) for a, b in combinations(texts, 2)]

    avg_bleu_score = np.mean(bleu_scores)

    return avg_bleu_score

def get_sbert_similarity(texts):

        # Create the model
        # model = SentenceTransformer("all-mpnet-base-v2")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Get the embeddings
        embeddings = model.encode(texts)

        # Calculate the cosine similarity
        similarity_matrix = cosine_similarity(embeddings)

        # Get the upper triangle
        upper_matrix = np.triu_indices_from(similarity_matrix, k=1)

        # Get the average similarity
        avg_similarity = np.mean(similarity_matrix[upper_matrix])

        return avg_similarity

def get_char_length(texts):

    # Get the character count
    char_count = [len(text) for text in texts]

    # Get the average character count
    avg_char_length = np.mean(char_count)

    return avg_char_length

def get_word_length(texts):

    # Get the word count
    word_count = [len(text.split()) for text in texts]

    # Get the average word count
    avg_word_length = np.mean(word_count)

    return avg_word_length

def get_token_length(texts):

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    token_count = [len(encoding.encode(text)) for text in texts]

    avg_token_count = np.mean(token_count)

    return avg_token_count

# Create the output folder
os.makedirs(output_folder, exist_ok=True)

# Create a table
similarity = pd.DataFrame()

# Get the file names
for file_name in os.listdir(input_folder):

    # Split filename into parts
    file_name_parts = file_name.split(" - ")

    # Only include CSV files
    if not file_name.endswith(".csv"):
        continue

    # Filter by model
    # TODO: Need to flip this back to exclude the specified model
    # TODO: And comment this section out
    if file_name_parts[0] != "claude-3-opus-20240229":
        continue

    # Filter by agent
    if file_name_parts[1] != "chain_of_thought":
        continue

    # Filter by exam
    if file_name_parts[2] != "comprehensive-100":
        continue

    # Display a status update
    print(f"Processing {file_name}")

    # Load the responses
    file_path = f"{input_folder}/{file_name}"
    table = pd.read_csv(file_path)

    # Rename the columns
    table.rename(columns={"Problem Id": "Problem ID"}, inplace=True)

    # Group by each problem
    groups = table.groupby(["Model Name", "Agent Name", "Exam Name", "Temperature", "Problem ID"])

    # Loop through each problem/row
    for name, group in groups:

        # Display a status update
        print(f" - Processing {name}")

        # Get the responses
        responses = group["Text"].tolist()

        # Verify there are 10 responses
        if len(responses) != 10:
            print(f" - WARNING: Expected 10 responses, but got {len(responses)}")

        # Create a row
        row = {
            "Model Name": group["Model Name"].iloc[0],
            "Agent Name": group["Agent Name"].iloc[0],
            "Exam Name": group["Exam Name"].iloc[0],
            "Temperature": group["Temperature"].iloc[0],
            "Problem ID": group["Problem ID"].iloc[0],
            "Jaccard Similarity": get_jaccard_simiarity(responses),
            "BoW Similarity": get_bag_of_words_similarity(responses),
            "TF-IDF Similarity": get_tfidf_similarity(responses),
            "Levenshtein Distance": get_levenshtein_distance(responses),
            "BLEU Score": get_bleu_score(responses),
            "SBERT Similarity": get_sbert_similarity(responses),
            "Character Length": get_char_length(responses),
            "Word Length": get_word_length(responses),
            "Token Length": get_token_length(responses)}

        # Add the log to the logs table
        similarity = similarity._append(row, ignore_index=True)

# Save the logs
similarity.to_csv(f"{output_folder}/text-similarity.csv", index=False)
