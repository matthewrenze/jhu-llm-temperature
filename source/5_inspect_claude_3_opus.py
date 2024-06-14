import os
import numpy as np
import pandas as pd
import anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

system_prompt = """
You are an intelligent assistant.
Your task is to answer the following multiple-choice questions.
Think step-by-step through the problem to ensure you have the correct answer.
Then, you MUST answer the question using the following format 'Action: Answer("[choice]")'  
The parameter [choice] is the letter or number of the answer you want to select (e.g. "A", "B", "C", or "D")
For example, 'Answer("C")' will select choice "C" as the best answer.
The answer MUST ALWAYS be one of the available choices; it CANNOT be "None of the Above".
If you think the answer is "none of the above", then you MUST select the most likely answer.
"""

example_problem = """
Question: What is the capital of the state where Johns Hopkins University is located?
Choices:
  A: Baltimore
  B: Annapolis
  C: Des Moines
  D: Las Vegas
"""

example_solution = """
Thought: 
  Johns Hopkins University is located in Baltimore.
  Baltimore is a city located in the State of Maryland.
  The capital of Maryland is Annapolis.
  Therefore, the capital of the state where Johns Hopkins University is located is Annapolis.
  The answer is B: Annapolis.
Action: Answer("B")  
"""

user_prompt = """
Topic: Geography and Math
Question: What is the product of the number of letters contained in the name of the city where Iowa State University is located multiplied by the number of letters contained in the name of the state?
Choices:
    A: 16
    B: 20
    C: 24
    D: 32
"""

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

# Set parameters
temperatures = [0.0, 0.5, 1.0]
num_attempts = 3

results = pd.DataFrame(columns=["Temperature", "Similarity"])

for temperature in temperatures:

    texts = []

    for i in range(num_attempts):

        # Create the messages
        messages = [
            {"role": "user", "content": example_problem},
            {"role": "assistant", "content": example_solution},
            {"role": "user", "content": user_prompt}]

        client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"))

        api_response = client.messages.create(
            model="claude-3-opus-20240229",
            temperature=temperature,
            max_tokens=1000,
            system=system_prompt,
            messages=messages)

        response = api_response.content[0].text.replace("\n\n", "\n")

        texts.append(response)

        print(f"Response:\n{response}")
        print(f"Temperature: {temperature}")
        print(f"Input tokens: {api_response.usage.input_tokens}")
        print(f"Output tokens: { api_response.usage.output_tokens}")
        print("")

    similarity = get_tfidf_similarity(texts)

    results = results._append({
        "Temperature": temperature,
        "Similarity": similarity},
        ignore_index=True)


print(results)