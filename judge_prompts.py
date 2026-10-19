# -*- coding: utf-8 -*-
"""
Prompt templates for judge model to evaluate debate outcomes.
"""

# Judge's evaluation of the debate
judge_decision = """You are a judge responsible for making a final decision based on the debate history between two models. They have debated the following question and provided their reasoning.

**Question:** {{prompt}}

**Debate History:**
- **Model 1 initial opinion:** {{model1_response}}
- **Model 2 initial opinion:** {{model2_response}}
- **Model 1 feedback:** {{model1_feedback}}
- **Model 2 feedback:** {{model2_feedback}}
- **Model 1 final decision:** {{model1_final}}
- **Model 2 final decision:** {{model2_final}}

**Your Task:**
Based solely on the debate history provided above, make a final judgment. Do NOT make any independent judgment beyond what is argued in the debate. Respond with ONLY your final decision in one of these formats:
- If you agree with Model 1's conclusion: "{{model1_final}}"
- If you agree with Model 2's conclusion: "{{model2_final}}"
- If neither argument is convincing or you determine both/neither have merit: "Neither"

Do not provide any explanation or reasoning. Only respond with the answer in the exact format specified above."""

judge_prompts = {
    "judge_decision": judge_decision
}
