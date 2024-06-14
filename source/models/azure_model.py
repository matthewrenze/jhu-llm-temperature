import requests
import json
import os
from models.model import Model
from models.response import Response

class AzureModel(Model):

    def __init__(self, model_name, temperature, log):
        super().__init__(model_name, temperature, log)
        self.api_key = os.getenv(f"{model_name.upper().replace('-', '_')}_API_KEY")
        self.api_url = os.getenv(f"{model_name.upper().replace('-', '_')}_URL")

    def get_response(self, messages, num_choices=1):

        try:

            response = Response()

            api_url = self.api_url + "/v1/chat/completions"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Content-type": "application/json"}

            body = {
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens}

            for i in range(num_choices):
                api_response = requests.post(api_url, headers=headers, json=body)
                api_response_body = api_response.content.decode("utf-8")
                api_response_body = json.loads(api_response_body)
                api_response_content = api_response_body["choices"][0]["message"]["content"]
                response.choices.append(api_response_content + "\n")
                response.input_tokens = api_response_body["usage"]["prompt_tokens"]
                response.output_tokens += api_response_body["usage"]["completion_tokens"]

        except Exception as e:
            self.log.error(f"Error: {str(e)}")
            response.has_error = True
            response.text = f"Error: {str(e)}"

        finally:
            response.total_tokens = response.input_tokens + response.output_tokens
            return response