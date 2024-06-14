import re

# Get the answer from the action
def get_answer(action: str):
    agent_answer = None
    agent_answer_match = re.search(r'Answer\("([^"]*)"\)', action)
    if agent_answer_match is not None:
        agent_answer = agent_answer_match.group(1)
        agent_answer = agent_answer.strip()
        if agent_answer != "":
            agent_answer = agent_answer[0].upper()
    return agent_answer