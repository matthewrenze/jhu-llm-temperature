# The Effect of Sampling Temperature on Problem Solving in Large Language Models

## Abstract
In this research study, we empirically investigate the effect of sampling temperature on the performance of Large Language Models (LLMs) on various problem-solving tasks. 

We created a multiple-choice question-and-answer (MCQA) exam by randomly sampling problems from standard LLM benchmarks. Then, we used four popular LLMs with five prompt-engineering techniques to solve the MCQA problems while increasing the sampling temperature from 0.0 to 1.0. 

Despite anecdotal reports to the contrary, our empirical results indicate that changes in temperature in the range 0.0 to 1.0 do not have a statistically significant impact on LLM performance for problem-solving tasks. In addition, these results appear to hold regardless of the LLM, the prompt-engineering technique, or the problem domain. 

## Documents
- [Paper](http://arxiv.org/abs/2402.05201)

## Code
- [Source](source/) - contains all source code
- [Models](source/models) - contains the model-specific code
- [Prompts](source/agents) - contains LLM agent prompt code
- [Process](source/process/) - contains the data pre-processing scripts
- [Analyze](source/analyze/) - contains the data analysis scripts

## Data
- [Exams](data/exams/) - contains the test dataset
- [Results](data/results/) - contains the high-level test results
- [Details](data/details/) - contains the low-level test results
- [Responses](data/responses/) - contains the LLM response text
- [Logs](data/logs/) - contains the experiment event logs

## Analysis
- [Plots](plots/) - contains all data visualizations

## Notes
- See [Requirements.txt](source/requirements.txt) for a list of packages used in this experiment.
- [GitHub Copilot](https://github.com/features/copilot) was used in the creation of source code for this experiment.

