"""Sample model actions without accidentally training on inserted tool outputs."""
import json
from .schema import Trajectory
from .tools import execute, tool_schemas
from .tool_format import parse_tool_calls
from .rewards import reward

SYSTEM = "Solve the question. Use the provided tools when useful. Finish with <final>answer only</final>. Call at most four tools."


def append_segment(ids, mask, segment, generated):
    ids.extend(segment)
    mask.extend([generated] * len(segment))


def initial_prompt(tokenizer, task, tools_enabled, enable_python=False):
    question = task.question
    if task.documents and not tools_enabled:
        question += "\nContext:\n" + "\n\n".join(doc["title"] + ": " + doc["text"] for doc in task.documents)
    elif task.documents:
        question += "\nAvailable local document titles: " + "; ".join(doc["title"] for doc in task.documents)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    kwargs = {"tools": tool_schemas(enable_python)} if tools_enabled else {}
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs)


def tool_suffix(tokenizer, response, generated):
    # Qwen2.5 groups tool messages inside a user turn with <tool_response> tags.
    # Generated IDs remain unchanged; appended protocol and tool bytes are masked.
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    end = "" if generated and generated[-1] == im_end else "<|im_end|>"
    text = end + "\n<|im_start|>user\n<tool_response>\n" + json.dumps(response, ensure_ascii=False) + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    return tokenizer.encode(text, add_special_tokens=False)


def sample_trajectory(model, tokenizer, task, *, temperature=1.0, max_new_tokens=512,
                      max_tool_calls=4, max_sequence_length=4096, tools_enabled=True,
                      enable_python=False):
    import torch
    ids = initial_prompt(tokenizer, task, tools_enabled, enable_python)
    mask = [False] * len(ids)
    turns, calls, parsed_calls, final_text, truncated = [], 0, 0, "", False
    model.eval()  # Disable LoRA dropout; sampling and importance denominator must agree.
    for _ in range(max_tool_calls + 1):
        room = max_sequence_length - len(ids)
        if room <= 0:
            truncated = True
            break
        inputs = torch.tensor([ids], device=next(model.parameters()).device)
        generation = dict(max_new_tokens=min(max_new_tokens, room), do_sample=temperature > 0,
                          pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                          use_cache=True, repetition_penalty=1.0, no_repeat_ngram_size=0, num_beams=1,
                          temperature=1.0, top_p=1.0, top_k=0)
        if temperature > 0:
            # No top-p/top-k truncation: log probabilities in the objective match sampling.
            generation.update(temperature=temperature, top_p=1.0, top_k=0)
        else:
            generation.update(temperature=None, top_p=None, top_k=None)
        if tools_enabled:
            generation.update(stop_strings=["</tool_call>"], tokenizer=tokenizer)
        with torch.no_grad():
            result = model.generate(inputs, **generation)
        generated = result[0, len(ids):].tolist()
        append_segment(ids, mask, generated, True)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        final_text = text
        turn = {"assistant": text, "generated_tokens": len(generated), "tools": []}
        turns.append(turn)
        try:
            requests = parse_tool_calls(text) if tools_enabled else []
        except ValueError as exc:
            turn["parse_error"] = str(exc)
            break
        if not requests:
            truncated |= len(generated) == generation["max_new_tokens"] and (not generated or generated[-1] != tokenizer.eos_token_id)
            break
        results = []
        for request in requests:
            if calls >= max_tool_calls:
                turn["tool_limit_reached"] = True
                break
            calls += 1
            parsed_calls += 1
            output = execute(request, task.documents, enable_python)
            turn["tools"].append({"call": request, "response": output})
            results.append(output)
        if not results:
            break
        suffix = tool_suffix(tokenizer, results[0] if len(results) == 1 else results, generated)
        if len(ids) + len(suffix) >= max_sequence_length:
            truncated = True
            break
        append_segment(ids, mask, suffix, False)
    return Trajectory(task.id, ids, mask, final_text, turns, reward(task, final_text, calls, parsed_calls), truncated)


def batch_direct_trajectories(model, tokenizer, tasks, *, max_new_tokens=512, max_sequence_length=4096):
    """Greedy direct answers with left padding; retain the original per-task token budget."""
    import torch
    prompts = [initial_prompt(tokenizer, task, False) for task in tasks]
    trajectories = [None] * len(tasks)
    short = []
    for index, prompt in enumerate(prompts):
        if len(prompt) + max_new_tokens <= max_sequence_length:
            short.append(index)
        else:
            trajectories[index] = sample_trajectory(model, tokenizer, tasks[index], temperature=0,
                tools_enabled=False, max_new_tokens=max_new_tokens, max_sequence_length=max_sequence_length)
    if not short:
        return trajectories
    model.eval()
    width = max(len(prompts[index]) for index in short)
    device = next(model.parameters()).device
    inputs = torch.full((len(short), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(inputs)
    for row, index in enumerate(short):
        prompt = prompts[index]
        inputs[row, -len(prompt):] = torch.tensor(prompt, dtype=torch.long, device=device)
        attention[row, -len(prompt):] = 1
    with torch.no_grad():
        outputs = model.generate(inputs, attention_mask=attention, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            use_cache=True, repetition_penalty=1.0, no_repeat_ngram_size=0, num_beams=1,
            temperature=None, top_p=None, top_k=None)
    for row, index in enumerate(short):
        generated = outputs[row, width:].tolist()
        if tokenizer.eos_token_id in generated:
            generated = generated[:generated.index(tokenizer.eos_token_id) + 1]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        prompt = prompts[index]
        truncated = len(generated) == max_new_tokens and generated[-1] != tokenizer.eos_token_id
        trajectories[index] = Trajectory(tasks[index].id, prompt + generated,
            [False] * len(prompt) + [True] * len(generated), text,
            [{"assistant": text, "generated_tokens": len(generated), "tools": []}],
            reward(tasks[index], text, 0, 0), truncated)
    return trajectories
