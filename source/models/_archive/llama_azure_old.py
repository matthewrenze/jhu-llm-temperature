# Import libraries
import time
import urllib.request
import json
import ssl
import os
from models.response import Response

class LlamaAzure:

    def __init__(self, chat_model, temperature):
        self.chat_model = chat_model
        self.temperature = temperature

    # Define a function to get the experiments from the Open AI API
    def get_response(self, messages, num_choices=1):

        try:

            # Disable SSL verification
            ssl._create_default_https_context = ssl._create_unverified_context

            # Create the response
            response = Response()

            # Create the body of the request
            body = {
                "input_data": {
                    "input_string": messages,
                    "parameters": {
                        "temperature": self.temperature,
                        "max_tokens": 4096
                    }
                }
            }

            body = str.encode(json.dumps(body))

            api_url = f"https://{self.chat_model}-chat-1577.eastus.inference.ml.azure.com/score"
            api_key = "4Yh2EdVj0fUDcKQzrfFvf6Y7X9gRRO2t"
            headers = {
                "Content-Type": "application/json",
                "Authorization": ("Bearer " + api_key),
                "azureml-model-deployment": f"{self.chat_model}-chat-1577"}

            # Create the API request
            api_request = urllib.request.Request(api_url, body, headers)

            # Loop through each choice
            for i in range(num_choices):

                # Get the API response
                api_response = urllib.request.urlopen(api_request)
                api_response_body = api_response.read()
                api_response_body = json.loads(api_response_body)

                # Append the response to the choices
                response.choices.append(api_response_body["output"] + "\n")

                # Get the tokens
                response.input_tokens = 0
                response.output_tokens = 0
                response.total_tokens = 0

        except Exception as e:

            # Add the error message
            response.has_error = True
            response.text = f"Error: {str(e)}"

        finally:

            # Pause for a second if using GPT-4
            if self.chat_model == "gpt-4":
                time.sleep(1)

            return response