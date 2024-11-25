# The Effect of Sampling Temperature on Problem Solving in Large Language Models

## Abstract
In this research study, we empirically investigate the effect of sampling temperature on the performance of Large Language Models (LLMs) on various problem-solving tasks. 

We created a multiple-choice question-and-answer (MCQA) exam by randomly sampling problems from standard LLM benchmarks. Then, we used nine popular LLMs with five prompt-engineering techniques to solve the MCQA problems while increasing the sampling temperature from 0.0 to 1.6. 

Despite anecdotal reports to the contrary, our empirical results indicate that changes in temperature from 0.0 to 1.0 do not have a statistically significant impact on LLM performance for problem-solving tasks. In addition, these results appear to generalize across LLMs, prompt-engineering techniques, and problem domains. 

## Documents
- [Research paper](https://aclanthology.org/2024.findings-emnlp.432)
- [Research poster](https://matthewrenze.com/wp-content/uploads/posters/llm-temperature.pdf)
- [Presentation video](https://youtu.be/VvhpKAXe_Mc)
- [Presentation slides](https://matthewrenze.com/wp-content/uploads/presentations/llm-temperature.pdf)
- [Pre-print paper](https://arxiv.org/abs/2402.05201)

## Code
- [Source](source/) - contains all source code
- [Models](source/models) - contains the model-specific code
- [Prompts](source/agents) - contains LLM agent prompt code
- [Exams](source/exams/) - contains the code to load exams 

## Data
- [Exams](data/exams/) - contains the test dataset
- [Results](data/results/) - contains the high-level test results
- [Details](data/details/) - contains the low-level test results
- [Responses](data/responses/) - contains the LLM response text
- [Logs](data/logs/) - contains the experiment event logs

## Analysis
- [Plots](plots/) - contains all data visualizations

## Notes
- [Source](source/) contains all scripts for experiments, processing, and analysis
- See [Requirements.txt](source/requirements.txt) for a list of packages used in this experiment.
- [GitHub Copilot](https://github.com/features/copilot) was used in the creation of this experiment.

