import os
import time
from openai import AzureOpenAI
from models.model import Model
from models.response import Response

class OpenAIModel(Model):

    def __init__(self, model_name, temperature, log):
        super().__init__(model_name, temperature, log)
        self.api_base = os.getenv("AZURE_OPENAI_URL")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = "2024-02-01"

    def get_response(self, messages, num_choices=1):

        try:

            response = Response()

            client = AzureOpenAI(
                azure_endpoint=self.api_base,
                api_key=self.api_key,
                api_version=self.api_version)

            # NOTE: Need to use "model" for Open AI and "engine" for Azure AI
            api_response = client.chat.completions.create(
                model=self.model_name,
                # engine=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=num_choices)

            response.input_tokens = api_response.usage.prompt_tokens
            response.output_tokens = api_response.usage.completion_tokens
            response.total_tokens = api_response.usage.total_tokens

            for choice in api_response.choices:
                response.choices.append(choice.message.content + "\n")

        except Exception as e:
            self.log.error(f"Error: {str(e)}")
            response.has_error = True
            response.text = f"Error: {str(e)}"

        finally:

            # Pause for a second if using GPT-4
            if self.model_name == 'gpt-4' \
                    or self.model_name == 'gpt-4o':
                time.sleep(1)

            return response