# -*- coding: utf-8 -*-
"""
Prompt templates for single LLM response generation.
Handles both judgment and plausibility tasks.
"""

# Single-turn prompts for judgment tasks
judgment_single = """{{prompt}}"""

# Single-turn prompts for plausibility tasks
plausibility_single = """{{prompt}}"""

# More detailed version with reasoning
judgment_with_reasoning = """{{prompt}}

Please provide your answer and briefly explain your reasoning in 2-3 sentences."""

plausibility_with_reasoning = """{{prompt}}

Please provide your answer and briefly explain your reasoning in 2-3 sentences."""

prompts = {
    "judgment_single": judgment_single,
    "plausibility_single": plausibility_single,
    "judgment_with_reasoning": judgment_with_reasoning,
    "plausibility_with_reasoning": plausibility_with_reasoning,
}
