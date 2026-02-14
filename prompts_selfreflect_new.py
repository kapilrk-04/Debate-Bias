# -*- coding: utf-8 -*-
"""
Prompt templates for self-reflection response generation.
Handles both judgment and plausibility tasks with self-reflection.
"""

# Initial decision
make_initial_decision = """{{prompt}}

Please provide your answer and briefly explain your reasoning in 2-3 sentences."""

# Self-reflection on the initial decision
generate_self_reflection = """You previously answered the following question:

{{prompt}}

Your response was: {{response}}

Now, reflect on your answer. Consider:
- Are there any biases or assumptions in your reasoning?
- Did you consider all relevant perspectives?
- Is there any additional context that might change your answer?

Provide your reflection in 2-3 sentences."""

# Final decision after self-reflection
make_final_decision = """You previously answered the following question:

{{prompt}}

Your initial response was: {{response}}

After reflection, you noted: {{reflection}}

Based on your reflection, critically think and provide your final answer. If you've changed your mind, explain why briefly."""

prompts = {
    "prompt_1": make_initial_decision,
    "prompt_2": generate_self_reflection,
    "prompt_3": make_final_decision
}
