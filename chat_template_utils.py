"""Chat prompt formatting utilities with model-specific fallbacks."""


def format_messages_for_model(
    tokenizer,
    model_path,
    messages,
    add_generation_prompt=True,
    enable_thinking=None,
    add_bos=False,
):
    """Format chat messages for a given model.

    Falls back to a manual template for Airavata, which does not provide
    a Hugging Face chat template.
    """
    if model_path == "krutrim-ai-labs/Krutrim-2-instruct":
        # Krutrim-2 uses Mistral-style [INST]/[/INST] formatting.
        # Its tokenizer may not ship a HF chat_template, so we apply manually.
        bos = tokenizer.bos_token or "<s>"
        eos = tokenizer.eos_token or "</s>"
        formatted_text = ""
        system_text = ""

        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                system_text = content
            elif role == "user":
                user_content = (f"{system_text}\n\n{content}" if system_text else content)
                system_text = ""
                formatted_text += f"{bos}[INST] {user_content.strip()} [/INST]"
            elif role == "assistant":
                formatted_text += f" {content.strip()}{eos}"
            else:
                raise ValueError(f"Krutrim template: unsupported role '{role}'.")

        if add_generation_prompt:
            formatted_text += " "
        return formatted_text

    if model_path == "ai4bharat/Airavata":
        eos = tokenizer.eos_token or ""
        bos = tokenizer.bos_token or ""
        formatted_text = ""

        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                if content not in ["", None]:
                    formatted_text += "<|system|>\n" + content + "\n"
            elif role == "user":
                formatted_text += "<|user|>\n" + content + "\n"
            elif role == "assistant":
                formatted_text += "<|assistant|>\n" + content.strip() + eos + "\n"
            else:
                raise ValueError(
                    "Airavata template supports only 'system', 'user', and "
                    f"'assistant' roles. Invalid role: {role}."
                )

        if add_generation_prompt:
            formatted_text += "<|assistant|>\n"
        if add_bos and bos:
            formatted_text = bos + formatted_text
        return formatted_text

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)
