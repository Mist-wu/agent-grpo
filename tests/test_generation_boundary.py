"""Entry regression for single-turn and batched generation boundaries.

Uses the cached Qwen tokenizer and a tiny random CPU model. No weights are downloaded.
"""
import json
import os
import unittest
from dataclasses import asdict

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


def _load_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', revision=REVISION, local_files_only=True)
    except OSError:
        return None


TOKENIZER = _load_tokenizer()


@unittest.skipUnless(TOKENIZER is not None, 'cached Qwen tokenizer is not available offline')
class GenerationBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from agent_grpo.schema import Task
        torch.set_num_threads(1)
        cls.torch = torch
        cls.tasks = [Task(str(i), 'gsm8k', 'Probe ' + label, '2', 'test')
                     for i, label in enumerate(['alpha', 'beta', 'gamma', 'delta'])]
        cls.tokenizer = TOKENIZER
        cls.model = _Scripted(cls.tokenizer, cls.tasks)

    def test_single_and_batch_entries_agree_on_boundaries(self):
        self._agree()
        original = self.tokenizer.pad_token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self._agree()
        finally:
            self.tokenizer.pad_token = original

    def test_single_tool_suffix_matches_native_template(self):
        from agent_grpo.rollout import tool_suffix
        tokenizer = self.tokenizer
        call = '<tool_call>\n{"name":"calculator","arguments":{"expression":"1+1"}}\n</tool_call>'
        payload = [{'result': '二'}, {'result': '三'}]
        base = [{'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': call}]
        prefix = tokenizer.apply_chat_template(base, tokenize=False)
        eos = tokenizer.eos_token_id
        for count in (1, 2):
            native = tokenizer.apply_chat_template(
                base + [{'role': 'tool', 'content': json.dumps(value, ensure_ascii=False)} for value in payload[:count]],
                tokenize=False, add_generation_prompt=True)
            for ended in (False, True):
                end = '<|im_end|>' if ended else ''
                stem = prefix.removesuffix('<|im_end|>\n') + end
                expected = tokenizer.encode(native[len(stem):], add_special_tokens=False)
                value = payload[:count] if count == 2 else payload[0]
                got = tool_suffix(tokenizer, value, [eos] if ended else tokenizer.encode(call, add_special_tokens=False))
                if count == 1:
                    self.assertEqual(got, expected)
                else:
                    self.assertNotEqual(got, expected)

    def _agree(self):
        from agent_grpo.rollout import batch_direct_trajectories, batch_tool_trajectories, initial_prompt, sample_trajectory
        tokenizer = self.tokenizer
        for tools, batched in ((True, batch_tool_trajectories), (False, batch_direct_trajectories)):
            limit = max(len(initial_prompt(tokenizer, task, tools)) for task in self.tasks) + 100
            args = dict(max_new_tokens=30, max_sequence_length=limit)
            single = [sample_trajectory(self.model, tokenizer, task, temperature=0, tools_enabled=tools, **args)
                      for task in self.tasks]
            grouped = batched(self.model, tokenizer, self.tasks, **args)
            self.assertEqual([asdict(item) for item in single], [asdict(item) for item in grouped])
            for item in grouped:
                self.assertEqual(len(item.token_ids), len(item.generated_mask))
                self.assertEqual(sum(item.generated_mask), sum(turn['generated_tokens'] for turn in item.turns))
                self.assertTrue(all(token != tokenizer.pad_token_id or tokenizer.pad_token_id == tokenizer.eos_token_id
                                     for token, generated in zip(item.token_ids, item.generated_mask) if generated))
        tight = len(initial_prompt(tokenizer, self.tasks[2], True)) + 2
        sample = sample_trajectory(self.model, tokenizer, self.tasks[2], temperature=0, max_new_tokens=10, max_sequence_length=tight)
        batch = batch_tool_trajectories(self.model, tokenizer, [self.tasks[2]], max_new_tokens=10, max_sequence_length=tight)[0]
        self.assertEqual(asdict(sample), asdict(batch))
        self.assertTrue(batch.truncated)
        self.assertEqual(sum(batch.generated_mask), 2)


class _Scripted:
    """Build after the tokenizer import succeeds; instantiated in setUpClass."""

    def __new__(cls, tokenizer, tasks):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id

        def encode(text):
            return tokenizer.encode(text, add_special_tokens=False)

        call = '<tool_call>\n{"name":"calculator","arguments":{"expression":"1+1"}}\n</tool_call>'

        class Model(Qwen2ForCausalLM):
            def generate(self, inputs, **kwargs):
                width = inputs.shape[1]
                scripts = []
                for ids in inputs.tolist():
                    prompt = tokenizer.decode(ids)
                    if 'Probe alpha' in prompt:
                        script = encode('<final>2</final>') + [eos] if '<tool_response>\n{' in prompt else encode(call) + [eos]
                    elif 'Probe beta' in prompt:
                        script = [eos]
                    elif 'Probe gamma' in prompt:
                        script = encode(' 7') * 80
                    else:
                        script = encode('<tool_call>') + encode('x') * 80
                    scripts.append(script)

                def force(ids, scores):
                    index = ids.shape[1] - width
                    scores.fill_(-float('inf'))
                    for row, script in enumerate(scripts):
                        scores[row, script[index] if index < len(script) else eos] = 0
                    return scores

                return super().generate(inputs, logits_processor=[force], **kwargs)

        torch.set_num_threads(1)
        return Model(Qwen2Config(vocab_size=len(tokenizer), hidden_size=8, intermediate_size=16,
                                 num_attention_heads=1, num_key_value_heads=1, num_hidden_layers=1,
                                 max_position_embeddings=1024, eos_token_id=eos, pad_token_id=pad))


if __name__ == '__main__':
    unittest.main()
