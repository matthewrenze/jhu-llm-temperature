# The Effect of Sampling Temperature on Problem-Solving in Large Language Models

**Author:** Matthew Renze  
**Class:** EN.705.742  
**Date:** 2023-12-08  

## Abstract
In this research study, we empirically investigate the relationship between sampling temperature and the performance of Large Language Models (LLMs) on various problem-solving tasks.

We created a multiple-choice question-and-answer (MCQA) exam by randomly sampling problems from standard LLM benchmarks. Then, we used four popular LLMs with five prompt-engineering techniques to solve the MCQA problems while increasing the sampling temperature from 0.0 to 1.0.

Despite anecdotal reports to the contrary, our empirical results indicate that temperature does not have a statistically significant impact on LLM performance for problem-solving tasks. In addition, these results appear to hold regardless of the LLM, the prompt-engineering technique, or the problem domain. 

These results have practical implications for AI systems engineers using LLMs to create automated AI systems. In addition, our results provide more general insight for AI researchers on the role of temperature-based sampling in model hallucination and solution-space search.

## Documents
- [Research paper](research-paper.pdf)

## Code
- [Source](source/) - contains all source code
- [Models](source/models) - contains the model-specific code
- [Prompts](source/agents) - contains LLM agent prompt code
- [Process](source/process/) - contains the data pre-processing scripts
- [Analyze](source/analyze/) - contains the data analysis scripts

## Data
- [Exams](exams/) - contains the test dataset
- [Results](results/) - contains the high-level test results
- [Details](details/) - contains the low-level test results
- [Logs](logs/) - contains the experiment event logs


## Analysis
- [Plots](plots/) - contains all data visualizations

