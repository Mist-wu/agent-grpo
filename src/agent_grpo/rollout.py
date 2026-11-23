"""Sample model actions without accidentally training on inserted tool outputs."""
from collections import defaultdict
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
    # Render only the continuation. Re-tokenizing the sampled prefix can change action IDs.
    base = [{"role": "user", "content": ""}]
    prefix = tokenizer.apply_chat_template(base, tokenize=False, add_generation_prompt=True)
    messages = base + [{"role": "assistant", "content": ""}]
    responses = response if isinstance(response, list) else [response]
    for value in responses:
        messages.append({"role": "tool", "content": json.dumps(value, ensure_ascii=False)})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    suffix = rendered[len(prefix):]
    if generated and generated[-1] == tokenizer.eos_token_id:
        suffix = suffix.removeprefix(tokenizer.eos_token)
    return tokenizer.encode(suffix, add_special_tokens=False)


def _cut_at_eos(generated, eos_id):
    """Keep the first EOS and drop batch padding that follows it."""
    if eos_id in generated:
        return generated[:generated.index(eos_id) + 1]
    return generated


def _new_state(ids):
    return {"ids": ids, "mask": [False] * len(ids), "turns": [], "calls": 0,
            "final_text": "", "truncated": False, "active": True}


def _finish_turn(state, task, tokenizer, generated, turn_budget, *,
                 tools_enabled, max_tool_calls, max_sequence_length, enable_python):
    append_segment(state["ids"], state["mask"], generated, True)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    state["final_text"] = text
    turn = {"assistant": text, "generated_tokens": len(generated), "tools": []}
    state["turns"].append(turn)
    hit_budget = len(generated) == turn_budget and (not generated or generated[-1] != tokenizer.eos_token_id)
    try:
        requests = parse_tool_calls(text) if tools_enabled else []
    except ValueError as exc:
        turn["parse_error"] = str(exc)
        requests = []
    if not requests:
        state["truncated"] |= hit_budget
        state["active"] = False
        return
    results = []
    for request in requests:
        if state["calls"] >= max_tool_calls:
            turn["tool_limit_reached"] = True
            break
        state["calls"] += 1
        output = execute(request, task.documents, enable_python)
        turn["tools"].append({"call": request, "response": output})
        results.append(output)
    if not results:
        state["active"] = False
        return
    suffix = tool_suffix(tokenizer, results, generated)
    if len(state["ids"]) + len(suffix) >= max_sequence_length:
        state["truncated"] = True
        state["active"] = False
    else:
        append_segment(state["ids"], state["mask"], suffix, False)


def _to_trajectory(task, state):
    return Trajectory(task.id, state["ids"], state["mask"], state["final_text"], state["turns"],
                      reward(task, state["final_text"], state["calls"], state["calls"]), state["truncated"])


def sample_trajectory(model, tokenizer, task, *, temperature=1.0, max_new_tokens=512,
                      max_tool_calls=4, max_sequence_length=4096, tools_enabled=True,
                      enable_python=False):
    return sample_trajectories(model, tokenizer, [task], temperature=temperature, max_new_tokens=max_new_tokens,
                               max_tool_calls=max_tool_calls, max_sequence_length=max_sequence_length,
                               tools_enabled=tools_enabled, enable_python=enable_python)[0]


def batch_direct_trajectories(model, tokenizer, tasks, *, max_new_tokens=512, max_sequence_length=4096):
    return sample_trajectories(model, tokenizer, tasks, tools_enabled=False,
                               max_new_tokens=max_new_tokens, max_sequence_length=max_sequence_length)


def sample_trajectories(model, tokenizer, tasks, *, temperature=0, max_new_tokens=512,
                        max_tool_calls=4, max_sequence_length=4096, tools_enabled=True, enable_python=False):
    """Batch model turns while keeping tool execution and termination per trajectory."""
    import torch
    from transformers import StoppingCriteria, StopStringCriteria

    tool_end = StopStringCriteria(tokenizer, ["</tool_call>"]) if tools_enabled else None

    class FinishLengths(StoppingCriteria):
        def __init__(self, width, budgets, device):
            self.width = width
            self.budgets = torch.tensor(budgets, device=device)
            self.lengths = torch.zeros(len(budgets), dtype=torch.long, device=device)

        def __call__(self, input_ids, scores, **kwargs):
            count = input_ids.shape[1] - self.width
            finished = (input_ids[:, -1] == tokenizer.eos_token_id) | (count >= self.budgets)
            if tool_end is not None:
                finished |= tool_end(input_ids, scores)
            self.lengths[(self.lengths == 0) & finished] = count
            return finished

    states = [_new_state(initial_prompt(tokenizer, task, tools_enabled, enable_python)) for task in tasks]
    model.eval()
    device = next(model.parameters()).device
    for _ in range(max_tool_calls + 1):
        batches = defaultdict(list)
        for index, state in enumerate(states):
            if not state["active"]:
                continue
            room = max_sequence_length - len(state["ids"])
            if room <= 0:
                state.update(active=False, truncated=True)
            else:
                budget = min(max_new_tokens, room)
                # Keep padded direct-answer batches within their original context limit.
                batches[None if tools_enabled else budget].append((index, budget))
        if not batches:
            break
        for batch in batches.values():
            active, budgets = zip(*batch)
            width = max(len(states[index]["ids"]) for index in active)
            inputs = torch.full((len(active), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
            attention = torch.zeros_like(inputs)
            for row, index in enumerate(active):
                ids = states[index]["ids"]
                inputs[row, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=device)
                attention[row, -len(ids):] = 1
            finish = FinishLengths(width, budgets, device)
            with torch.no_grad():
                outputs = model.generate(inputs, attention_mask=attention, max_new_tokens=max(budgets),
                    do_sample=temperature > 0, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    use_cache=True, repetition_penalty=1.0, no_repeat_ngram_size=0, num_beams=1,
                    temperature=temperature if temperature > 0 else None,
                    top_p=1.0 if temperature > 0 else None, top_k=0 if temperature > 0 else None,
                    stopping_criteria=[finish])
            lengths = finish.lengths.tolist()
            max_generated = outputs.shape[1] - width
            for row, index in enumerate(active):
                length = lengths[row] or min(budgets[row], max_generated)
                generated = _cut_at_eos(outputs[row, width:width + length].tolist(), tokenizer.eos_token_id)
                _finish_turn(states[index], tasks[index], tokenizer, generated, budgets[row],
                             tools_enabled=tools_enabled, max_tool_calls=max_tool_calls,
                             max_sequence_length=max_sequence_length, enable_python=enable_python)
    return [_to_trajectory(task, state) for task, state in zip(tasks, states)]


batch_tool_trajectories = sample_trajectories
