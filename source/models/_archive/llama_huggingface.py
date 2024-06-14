# Import libraries
import time
import requests
import json
import os
from transformers import AutoTokenizer
from models.response import Response

class LlamaHuggingface():

    def __init__(self, chat_model, temperature):
        self.chat_model = chat_model
        self.temperature = temperature

    # Define a function to get the experiments from the Open AI API
    def get_response(self, messages, num_choices=1):

        try:

            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf",
                token="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")

            # Create the response
            response = Response()

            # Convert the messages into formatted string
            text = tokenizer.apply_chat_template(messages, tokenize=False)

            # Get the input tokens
            response.input_tokens = len(tokenizer.tokenize(text))

            # Create the body of the request
            body = {
                "inputs": text,
                "parameters": {
                    "temperature": self.temperature,
                    "max_new_tokens": 500}}

            # Create the parameters for the API request
            api_url = f"https://z1dlb0e5ni3anklj.us-east-1.aws.endpoints.huggingface.cloud"
            api_key = "cmsdXoMFvqjgQHmFsiavvxKsviSpXpIuryqbjjUsTDtZXCoskfcwjDASgsrkFVVwEiRLJidzXWIJPgVrKJPwZvKbXZnqCPfqiJmwXBQwrMlWOOMpZXJJvOBhZqREGbEB"
            headers = {
                "Content-Type": "application/json",
                "Authorization": ("Bearer " + api_key)}


            # Loop through each choice
            for i in range(num_choices):

                # Get the API response
                api_response = requests.post(api_url, headers=headers, json=body)

                api_response_body = api_response.content.decode("utf-8")
                api_response_body = json.loads(api_response_body)
                api_response_text = api_response_body[0]["generated_text"]

                # Append the response to the choices
                response.choices.append(api_response_text + "\n")

                # Increment the output tokens
                response.output_tokens += len(tokenizer.tokenize(api_response_text))

        except Exception as e:

            # Add the error message
            response.has_error = True
            response.text = f"Error: {str(e)}"

        finally:

            # Get the total tokens
            response.total_tokens = response.input_tokens + response.output_tokens

            return response