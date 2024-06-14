# Add the baseline results to the other agent types
# TODO: This should be made into a reusable function and shared across plotting scripts
import pandas as pd

model_titles = {
    "gpt-35-turbo": "GPT-3.5 Turbo",
    "gpt-4": "GPT-4",
    "gpt-4o": "GPT-4o",
    "llama-2-7b-chat": "Llama 2 7B",
    "llama-2-70b-chat": "Llama 2 70B",
    "mistral-large": "Mistral Large",
    "cohere-command-r-plus": "Command R+",
    "gemini-1.0-pro": "Gemini Pro 1.0",
    "gemini-1.5-pro-preview-0409": "Gemini Pro 1.5",
    "claude-3-opus-20240229": "Claude 3 Opus"}

exam_titles = {
    "comprehensive-100": "Comprehensive",
    "aqua-rat-100": "AQUA-RAT",
    "logiqa-en-100": "LogiQA",
    "lsat-ar-100": "LSAT-AR",
    "lsat-lr-100": "LSAT-LR",
    "lsat-rc-100": "LSAT-RC",
    "sat-en-100": "SAT-English",
    "sat-math-100": "SAT-Math",
    "arc-challenge-test-100": "ARC Challenge",
    "hellaswag_val-100": "Hellaswag",
    "medmcqa-dev-100": "MedMCQA"}

agent_titles = {
    "baseline": "Baseline",
    "domain_expert": "Domain Expert",
    "self_recitation": "Self-Recitation",
    "chain_of_thought": "Chain of Thought",
    "composite": "Composite"}

def get_model_title(model_name: str):
    return model_titles[model_name]

def set_model_titles(results: pd.DataFrame):
    results["Model Title"] = results["Model Name"].replace(model_titles)
    return results

def get_exam_title(exam_name: str):
    return exam_titles[exam_name]

def set_exam_titles(results: pd.DataFrame):
    results["Exam Title"] = results["Exam Name"].replace(exam_titles)
    return results

def get_agent_title(agent_name: str):
    return agent_titles[agent_name]

def set_agent_titles(results: pd.DataFrame):
    results["Agent Title"] = results["Agent Name"].replace(agent_titles)
    return results

def sort_by_model(results: pd.DataFrame):
    results["Model Title"] = pd.Categorical(results["Model Title"], [
        "Llama 2 7B",
        "Llama 2 70B",
        "GPT-3.5 Turbo",
        "Gemini Pro 1.0",
        "Command R+",
        "Mistral Large",
        "Gemini Pro 1.5",
        "GPT-4",
        "Claude 3 Opus"])

    return results

def sort_by_agent(results: pd.DataFrame):
    results["Agent Title"] = pd.Categorical(results["Agent Title"], [
        "Baseline",
        "Domain Expert",
        "Self-Recitation",
        "Chain of Thought",
        "Composite"])
    return results

def sort_by_exam(results: pd.DataFrame):
    results["Exam Title"] = pd.Categorical(results["Exam Title"], [
        # "Comprehensive",
        "AQUA-RAT",
        "ARC Challenge",
        "Hellaswag",
        "LogiQA",
        "LSAT-AR",
        "LSAT-LR",
        "LSAT-RC",
        "MedMCQA",
        "SAT-English",
        "SAT-Math"])
    return results