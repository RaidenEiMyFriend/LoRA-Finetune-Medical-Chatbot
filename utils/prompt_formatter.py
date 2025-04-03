def formatting_prompts_func(examples, tokenizer):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        messages = [
            {"role": "system", "content": "You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. Please answer the following medical question."},
            {"role": "user", "content": f"Question: {input}"},
            {"role": "assistant", "content": f"Reasoning:\n{cot}\nFinal Answer:\n{output}"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}
