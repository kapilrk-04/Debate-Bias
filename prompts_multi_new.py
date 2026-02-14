# -*- coding: utf-8 -*-
"""
Prompt templates for multi-agent debate response generation.
Handles both judgment and plausibility tasks with collaborative reasoning.
"""

# Initial decision from each agent
make_initial_decision = """{{prompt}}

Please provide your answer and briefly explain your reasoning in 2-3 sentences."""

# Provide feedback on other agent's response
give_feedback = """You are discussing the following question with another agent:

{{prompt}}

Your response: {{your_response}}
Other agent's response: {{other_response}}

Please provide feedback on the other agent's response. Consider:
- Do you agree or disagree with their reasoning?
- What aspects did they consider that you might have missed?
- Are there any flaws in their reasoning?

Respond in 2-3 sentences."""

# Make final decision after discussion
make_final_decision = """You are discussing the following question with another agent:

{{prompt}}

Your initial response: {{your_response}}
Other agent's initial response: {{other_response}}
Your feedback to the other agent: {{your_feedback}}
Other agent's feedback to you: {{other_feedback}}

Based on this discussion, critically think and provide your final answer. If you've changed your mind, explain why briefly.
If the task asks for a specific format (e.g., "Answer with just the identity name" or "Answer either 1 or 2"), make sure to follow that format in your final answer."""

prompts = {
    "prompt_1": make_initial_decision,
    "prompt_2": give_feedback,
    "prompt_3": make_final_decision
}
