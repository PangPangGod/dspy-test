import dspy

dspy.configure(lm=dspy.OpenAI(model="gpt-3.5-turbo"))

class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

generate_answer = dspy.ChainOfThought(BasicQA)

# Call the predictor on a particular input alongside a hint.
question="From Heidegger's perspective, why do I live?"
pred = generate_answer(question=question)

print(pred)