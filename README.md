# Comparative Analysis of Model Accuracy Using Mixed-precision Techniques at Various Levels of Sparsity

This repository contains the code and experiments conducted for the paper titled **"Comparative Analysis of Model Accuracy Using Mixed-precision Techniques at Various Levels of Sparsity."** This study evaluates the impact of mixed-precision techniques on the performance of LLM when applied at various levels of sparsity induced by pruning.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Experimentation](#experimentation)
  - [Pruning Methodology](#pruning-methodology)
  - [Mixed-precision Implementation](#mixed-precision-implementation)
- [Results](#results)

## Introduction

In this study, we examine the impact of applying pruning and mixed-precision techniques to large language models, specifically focusing on their effects on model accuracy across different sparsity levels. By combining these two techniques, we aim to optimize the computational efficiency of large models without significantly compromising their performance.

## Prerequisites

To run the experiments and use the code in this repository, you'll need the following:

- Python 3.8 or higher
- PyTorch 2.4.0
- CUDA 12.1
- NVIDIA GPU (e.g., RTX 4090)
- Huggingface Transformers library (version 4.43.4)
- GLUE Benchmark dataset

## Experimentation

### Pruning Methodology

In the experiments, pruning is applied incrementally, increasing sparsity levels from 50% to 99%. The model's performance is evaluated at each sparsity level to analyze the impact of pruning.

### Mixed-precision Implementation

Mixed-precision computation is implemented using PyTorch’s Automatic Mixed Precision (AMP). Critical operations such as weight updates are computed at 32-bit precision (float32), while other operations are performed at 16-bit precision (float16).

## Results

The results of our experiments show that:

- Model accuracy generally decreases with increasing sparsity.
- Mixed-precision computation significantly reduces training time and memory usage without substantially affecting model accuracy.
- At very high levels of sparsity, the remaining model parameters are often sufficient to maintain core model functions, suggesting that the impact of mixed-precision is minimal.

