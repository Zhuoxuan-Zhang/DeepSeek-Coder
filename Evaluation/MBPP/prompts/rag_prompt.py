def construct_rag_prompt(retrieved_context):
    # it is a rag prompt for code generation models. retrieved_context is the code retrieved from the database
    # ask the model to generate the code for the given context if it is relevant
    rag_prompt = "\n\nThe following code is a reference for your code generation: \n"
    rag_prompt += "\n>>> Reference Code:\n```python\n{}\n```\n".format(
        retrieved_context)
    rag_prompt += "\nExamine if the reference code is useful for the task. If useful, use the reference to generate the code. If not useful, ignore the reference and generate the code from scratch.\n"
    return rag_prompt
