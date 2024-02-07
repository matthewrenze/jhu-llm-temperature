# Import libraries
import time
import requests
import json
import os
from models.response import Response

# NOTE: This is the Anyscale API implementation
# NOTE: I can either use the REST (below) or OpenAI API flavors
# NOTE: See: https://app.endpoints.anyscale.com/
class Llama():

    def __init__(self, chat_model, temperature):
        self.chat_model = chat_model
        self.temperature = temperature
        self.api_key = os.getenv("ANYSCALE_API_KEY")

    # Define a function to get the results from the Open AI API
    def get_response(self, messages, num_choices=1):

        try:

            # Create the response
            response = Response()

            # Set the API parameters
            api_url = "https://api.endpoints.anyscale.com/v1/chat/completions"

            # Set the model name
            if self.chat_model == "llama-2-7b":
                model_name = "meta-llama/Llama-2-7b-chat-hf"
            elif self.chat_model == "llama-2-70b":
                model_name = "meta-llama/Llama-2-70b-chat-hf"
            else:
                raise Exception(f"Error: Invalid model name: {self.chat_model}")

            # Create the headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"}

            # Create the body
            body = {
                "model": model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": 4096,
                "seed": 42}

            # Loop through each choice
            for i in range(num_choices):

                # Get the API response
                api_response = requests.post(api_url, headers=headers, json=body)

                # Get the API response body (json)
                api_response_body = api_response.content.decode("utf-8")
                api_response_body = json.loads(api_response_body)

                # Get the LLM's response
                api_response_content = api_response_body["choices"][0]["message"]["content"]

                # Append the response to the choices
                response.choices.append(api_response_content + "\n")

                # Set the input tokens (don't increment)
                response.input_tokens = api_response_body["usage"]["prompt_tokens"]

                # Increment the output tokens
                response.output_tokens += api_response_body["usage"]["completion_tokens"]

        except Exception as e:

            # Add the error message
            response.has_error = True
            response.text = f"Error: {str(e)}"

        finally:

            # Get the total tokens
            response.total_tokens = response.input_tokens + response.output_tokens

            return response