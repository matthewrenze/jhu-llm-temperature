from agents.baseline_agent import BaselineAgent
from agents.domain_expert_agent import DomainExpertAgent
from agents.self_recitation_agent import SelfRecitationAgent
from agents.chain_of_thought_agent import ChainOfThoughtAgent
from agents.composite_agent import CompositeAgent
from models.gpt import Gpt
from models.llama import Llama

class AgentFactory:
    def __init__(self):
        pass

    def create_agent(self, model_type, agent_type, temperature, expertise, num_choices, log):

        # Create the model
        model = None

        if model_type == "gpt-35-turbo" or model_type == "gpt-4":
            model = Gpt(model_type, temperature)

        elif model_type == "llama-2-7b" or model_type == "llama-2-70b":
            model = Llama(model_type, temperature)

        else:
            raise Exception(f"Unknown model type: {model_type}")

        # Create the agent
        if agent_type == "baseline":
            return BaselineAgent(model, expertise, num_choices, log)

        elif agent_type == "domain_expert":
            return DomainExpertAgent(model, expertise, num_choices, log)

        elif agent_type == "self_recitation":
            return SelfRecitationAgent(model, expertise, num_choices, log)

        elif agent_type == "chain_of_thought":
            return ChainOfThoughtAgent(model, expertise, num_choices, log)

        elif agent_type == "composite":
            return CompositeAgent(model, expertise, num_choices, log)

        else:
            raise Exception(f"Unknown agent type: {agent_type}")
        