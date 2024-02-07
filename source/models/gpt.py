# Import libraries
import openai
import time
from models.response import Response

class Gpt():

    def __init__(self, chat_model, temperature):
        self.chat_model = chat_model
        self.temperature = temperature

    # Define a function to get the results from the Open AI API
    def get_response(self, messages, num_choices=1):

        try:

            # Create the response
            response = Response()

            # Call the Open AI API
            # NOTE: Need to use "model" for Open AI and "engine" for Azure AI
            api_response = openai.ChatCompletion.create(
                model=self.chat_model,
                engine=self.chat_model,
                messages=messages,
                temperature=self.temperature,
                n=num_choices)

            # Get the tokens
            response.input_tokens = api_response["usage"]["prompt_tokens"]
            response.output_tokens = api_response["usage"]["completion_tokens"]
            response.total_tokens = api_response["usage"]["total_tokens"]

            # Get the content from multiple choices
            for choice in api_response["choices"]:
                response.choices.append(choice["message"]["content"] + "\n")

        except Exception as e:

            # Add the error message
            response.has_error = True
            response.text = f"Error: {str(e)}"

        finally:

            # Pause for a second if using GPT-4
            if self.chat_model == 'gpt-4':
                time.sleep(1)

            return response