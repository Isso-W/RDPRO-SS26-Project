"""Prompt templates reserved for optional guided generation.

The current implementation uses fixed templates and static checks. These
strings document the expected generation and review context for integrations
that need prompt text.
"""

CODER_PROMPT = """
Generate runnable PyTorch training, evaluation, inference, and experiment
driver files from Module 3 candidate configurations. Treat model_config as the
source of truth and tasks as explanatory context only.
"""

REVIEWER_PROMPT = """
Review generated code for required files, compile success, smoke-test success,
candidate sweep coverage, metric/task consistency, and finetune freezing.
"""
