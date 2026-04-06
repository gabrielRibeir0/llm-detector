You are a science and engineering assistant.

Here are X questions in JSON format:
{questions}

Instructions:
1. For each question, write an 80-120 word EXPLANATION about the topic of the question, not just the answer
2. The explanation should provide context, examples, and detail — not just name the answer
3. No non-textual characters (markdown, emojis, etc.)
4. Return ONLY valid JSON, no preamble or explanations
5. Format: {{"Q1": "explanation", "Q2": "explanation", ...}}
6. Use the same keys (Q1, Q2, etc.) as the input

Return only the JSON with the explanations:
