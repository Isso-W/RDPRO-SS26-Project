"""Prompt text placeholders for future LLM-backed Module 4 variants.

The current implementation is deterministic and template-based. These strings
document the intended Coder -> Executor -> Reviewer loop without requiring an
LLM dependency.
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

