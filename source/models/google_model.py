import os
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part
import time
from models.model import Model
from models.response import Response

class GoogleModel(Model):

    def __init__(self, model_name: str, temperature, log):
        super().__init__(model_name, temperature, log)
        self.region = os.getenv("GOOGLE_REGION")
        self.project_id = os.getenv("GOOGLE_PROJECT_ID")

    def get_response(self, messages, num_choices=1):

        try:

            response = Response()
            dialog = []

            for i, message in enumerate(messages):
                role = "model" if message["role"] == "assistant" else "user"
                part = Part.from_text(message["content"])
                content = Content(role=role, parts=[part])
                dialog.append(content)

            # NOTE: Gemini requires pair-wise prompt and response (i.e. no system prompt).
            # NOTE: So I need to create and insert an initial model response
            text = "I understand the instructions. Please proceed."
            parts = [Part.from_text(text)]
            content = Content(role="model", parts=parts)
            dialog.insert(1, content)

            old_messages = dialog[:-1]
            new_message = dialog[-1]
            new_text = new_message.parts[0].text

            vertexai.init(
                project=self.project_id,
                location=self.region)
            config = GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                candidate_count=1)  # Candidate count is always 1 for Gemini
            model = GenerativeModel(
                model_name=self.model_name,
                generation_config=config)

            for i in range(num_choices):

                chat = model.start_chat(
                    history=old_messages)

                api_response = chat.send_message(new_text)

                # Note: Set tokens first in case there is an error below
                response.input_tokens = api_response._raw_response.usage_metadata.prompt_token_count
                response.output_tokens += api_response._raw_response.usage_metadata.candidates_token_count
                response.total_tokens = response.input_tokens + response.output_tokens

                if api_response.candidates[0].finish_reason == "SAFETY":
                    raise Exception("Content safety filter was triggered.")

                if api_response.candidates[0].finish_reason == "MAX_TOKENS":
                    raise Exception("Maximum response length was exceeded.")

                candidate_text = api_response.candidates[0].content.parts[0].text
                candidate_text = candidate_text.replace("\n\n", "\n")
                response.choices.append(candidate_text + "\n")

                # NOTE: Gemini Pro 1.5 current has a quota of 5 requests per minute.
                if self.model_name == "gemini-1.5-pro-preview-0409":
                    time.sleep(15)

        except Exception as e:
            self.log.error(f"Error: {str(e)}")
            response.has_error = True
            response.text = f"Error: {str(e)}"

        finally:

            return response
