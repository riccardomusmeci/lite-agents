SUMMARIZE_MEMORY_PROMPT = """
Analyze the interaction steps and provide a concise memory summary.
Focus ONLY on what is essential for the next interaction.

STEPS:
{steps}

OUTPUT FORMAT:
SUMMARY: <1-2 sentences summarizing the goal and current state>
KEY_FACTS: <bullet points of new, useful information learned>
OUTCOME: <SUCCESS, FAILURE, or IN_PROGRESS>
"""