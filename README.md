# KGAFT

**Knowledge Graph-Driven Adaptive Multi-Agent Fine-Tuning Service for Domain-Specific LLM**

KGAFT is an automated domain-specific large language model fine-tuning framework driven by a knowledge graph and multi-agent collaboration. It aims to provide a low-barrier, high-reliability fine-tuning service for non-expert users by integrating dataset selection, requirement analysis, hyperparameter customization, iterative evaluation, and closed-loop optimization.

---

## Overview

Domain-specific fine-tuning of large language models often faces three major challenges:

- Redundant and low-quality training data
- Experience-dependent hyperparameter configuration
- Lack of automated quality assurance during model delivery

To address these issues, KGAFT proposes a knowledge graph-driven adaptive multi-agent system that supports:

- intelligent dataset filtering,
- automatic user requirement understanding,
- adaptive hyperparameter recommendation,
- iterative model evaluation,
- closed-loop fine-tuning optimization.

The system can dynamically adjust training strategies according to user expectations and hardware constraints.

---

## Framework

The KGAFT workflow consists of three core agents:

### 1. Data Selection Agent (`S2l/`)
Selects high-value training samples from large-scale domain datasets using small-model learning trajectory summarization.

### 2. Fine-Tuning Customization Agent (`Requirement_analysis_agent/`)
Parses natural-language user requirements and generates executable fine-tuning configurations using knowledge graph-enhanced reasoning.

### 3. Model Evaluation Agent (`Evaluation/`)
Evaluates intermediate fine-tuned models and determines whether iterative tuning should continue based on user target satisfaction.

---

## Repository Structure

```text
KGAFT/
├── Dataset/                        # Experimental datasets
├── Evaluation/                     # Model evaluation agent
├── Requirement_analysis_agent/     # Fine-tuning customization agent
├── S2l/                      # Data selection agent
├── workflow.py                     # End-to-end workflow orchestration
└── README.md