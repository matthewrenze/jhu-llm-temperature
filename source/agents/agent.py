class Agent:

    def __init__(self, model, expertise, num_choices, log):
        self.model = model
        self.expertise = expertise
        self.system_prompt_template = ""
        self.example_problem = ""
        self.example_solution = ""
        self.num_choices = num_choices
        self.log = log

    def request(self, global_messages):

        # Create the system prompt
        system_prompt = self.system_prompt_template
        system_prompt = system_prompt.replace("{{expertise}}", self.expertise)
        system_prompt = system_prompt.strip()

        # Create the user prompt
        user_prompt = "\n".join(m for m in global_messages)
        user_prompt = user_prompt.strip()

        # Create the messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.example_problem},
            {"role": "assistant", "content": self.example_solution},
            {"role": "user", "content": user_prompt}]

        # Get the response
        response = self.model.get_response(messages, self.num_choices)

        # If the response has an error, return the error message
        if response.has_error:
            return response

        # TESTING: Log the messages
        self.log.debug(f"Debug: Agent system prompt:\n{system_prompt}")
        self.log.debug(f"Debug: Agent user prompt:\n{user_prompt}")
        for i, choice in enumerate(response.choices):
            self.log.debug(f"Debug: Agent response {i}:\n{choice}")

        # Return the response
        return response