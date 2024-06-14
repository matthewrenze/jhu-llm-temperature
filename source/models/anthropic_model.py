import os
import anthropic
from models.model import Model
from models.response import Response

class AnthropicModel(Model):

    def __init__(self, model_name, temperature, log):
        super().__init__(model_name, temperature, log)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    def get_response(self, messages, num_choices=1):

        try:

            response = Response()

            system_message = messages.pop(0)

            for i in range(num_choices):

                client = anthropic.Anthropic(
                    api_key=self.api_key)

                api_response = client.messages.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    system=system_message["content"],
                    messages=messages)

                # Note: Set tokens first in case there is an error below
                response.input_tokens = api_response.usage.input_tokens
                response.output_tokens += api_response.usage.output_tokens

                if api_response.stop_reason == "max_tokens":
                    raise Exception("Maximum response length was exceeded.")

                response.text = api_response.content[0].text
                response.text = response.text.replace("\n\n", "\n")
                response.choices.append(response.text + "\n")

        except Exception as e:
            self.log.error(f"Error: {str(e)}")
            response.has_error = True
            response.text = f"Error: {str(e)}"

        finally:
            response.total_tokens = response.input_tokens + response.output_tokens
            return response