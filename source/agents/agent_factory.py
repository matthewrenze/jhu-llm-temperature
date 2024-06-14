from agents.baseline_agent import BaselineAgent
from agents.domain_expert_agent import DomainExpertAgent
from agents.self_recitation_agent import SelfRecitationAgent
from agents.chain_of_thought_agent import ChainOfThoughtAgent
from agents.composite_agent import CompositeAgent

class AgentFactory:
    def __init__(self):
        pass

    def create(self, model, agent_name, expertise, num_choices, log):

        # Create the agent
        if agent_name == "baseline":
            return BaselineAgent(model, expertise, num_choices, log)

        elif agent_name == "domain_expert":
            return DomainExpertAgent(model, expertise, num_choices, log)

        elif agent_name == "self_recitation":
            return SelfRecitationAgent(model, expertise, num_choices, log)

        elif agent_name == "chain_of_thought":
            return ChainOfThoughtAgent(model, expertise, num_choices, log)

        elif agent_name == "composite":
            return CompositeAgent(model, expertise, num_choices, log)

        else:
            raise Exception(f"Unknown agent type: {agent_name}")
        