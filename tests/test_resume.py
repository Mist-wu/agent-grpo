"""Stage recovery and truthful completion gates with controlled model outputs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from agent_grpo.schema import Task, Trajectory, write_jsonl
from agent_grpo.stages import collect, evaluate


class ResumeTests(unittest.TestCase):
    def test_collection_resume_skips_accepted_and_validates_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'train.jsonl'
            config = root / 'config.json'
            out = root / 'cold.jsonl'
            write_jsonl(data, [Task('a', 'gsm8k', 'Q', '2', 'train'), Task('b', 'gsm8k', 'Q2', '3', 'train')])
            config.write_text(json.dumps({'temperature': 1.0, 'group_size': 2, 'update_epochs': 2,
                                          'max_new_tokens': 10, 'max_sequence_length': 30, 'model': 'fixture'}))
            meta = {'config': json.loads(config.read_text()), 'data_sha256': 'abc',
                    'model_revision_resolved': 'fixture', 'git_commit': 'deadbeef',
                    'created_at': '2026-11-05T00:00:00+00:00'}

            def sample_batch(model, tokenizer, tasks, **kwargs):
                return [Trajectory(task.id, [1, 2], [False, True], '<final>2</final>', [], {'correctness': 1})
                        for task in tasks]

            with patch('agent_grpo.stages.model_and_tokenizer', return_value=(None, None)), \
                 patch('agent_grpo.stages.seed_all'), \
                 patch('agent_grpo.stages.metadata', side_effect=lambda *args: dict(meta)), \
                 patch('agent_grpo.stages.batch_tool_trajectories', side_effect=sample_batch) as sampling:
                collect(config, data, out, target=1)
                self.assertEqual(sampling.call_count, 1)
                collect(config, data, out, target=1)
                self.assertEqual(sampling.call_count, 1)
                record = json.loads(Path(str(out) + '.metadata.json').read_text())
                self.assertEqual(record['accepted'], 1)
                self.assertEqual(record['resume_count'], 1)
                meta['data_sha256'] = 'changed'
                with self.assertRaisesRegex(ValueError, 'data_sha256'):
                    collect(config, data, out, target=1)

    def test_batch_size_and_missing_eval_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'test.jsonl'
            config = root / 'config.json'
            out = root / 'eval.jsonl'
            write_jsonl(data, [Task('t1', 'gsm8k', 'Q', '2', 'test')])
            config.write_text(json.dumps({'temperature': 1.0, 'group_size': 2, 'update_epochs': 2,
                                          'max_new_tokens': 10, 'max_sequence_length': 30, 'model': 'fixture'}))
            with self.assertRaisesRegex(ValueError, 'batch_size'):
                collect(config, data, root / 'cold.jsonl', batch_size=0)
            with self.assertRaisesRegex(ValueError, 'batch_size'):
                evaluate(config, data, out, 'base-direct', batch_size=0)
            out.write_text('{"task_id": "t1"}\n')
            with patch('agent_grpo.stages.seed_all'), \
                 self.assertRaisesRegex(ValueError, 'provenance'):
                evaluate(config, data, out, 'base-direct', batch_size=1)

    def test_training_resume_checks_checkpoint_metadata_and_preserves_run_json(self):
        import torch
        from agent_grpo.training import training
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / 'train.jsonl'
            dev_path = root / 'dev.jsonl'
            config_path = root / 'config.json'
            ckpt = root / 'checkpoint-0001'
            out = root / 'run'
            ckpt.mkdir()
            out.mkdir()
            write_jsonl(train_path, [Task('tr1', 'gsm8k', 'Q', '2', 'train')])
            write_jsonl(dev_path, [Task('dv1', 'gsm8k', 'Q', '2', 'dev')])
            cfg = {'temperature': 1.0, 'group_size': 2, 'update_epochs': 1, 'steps': 1,
                   'learning_rate': 1e-5, 'dtype': 'float32', 'device': 'cpu', 'checkpoint_every': 1,
                   'max_new_tokens': 10, 'max_sequence_length': 32, 'questions_per_step': 1,
                   'kl_beta': 0.04, 'clip_epsilon': 0.2, 'model': 'fixture'}
            config_path.write_text(json.dumps(cfg))
            original_run = {'preserved': True}
            (out / 'run.json').write_text(json.dumps(original_run))
            (out / 'env.txt').write_text('original-env\n')
            model = torch.nn.Linear(2, 2)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            import random
            torch.save({'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(), 'step': 1,
                        'sampling_rng': random.Random(42).getstate(), 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': []}, ckpt / 'training-state.pt')
            meta = {'config': {**cfg, 'seed': 42, 'initial_adapter': 'adapter'}, 'data_sha256': 'same',
                    'model_revision_resolved': 'rev1'}
            (ckpt / 'metadata.json').write_text(json.dumps({**meta, 'data_sha256': 'different'}))
            with patch('agent_grpo.training.model_and_tokenizer', return_value=(model, None)), \
                 patch('agent_grpo.training.seed_all'), \
                 patch('agent_grpo.training.metadata', side_effect=lambda *args: dict(meta)):
                with self.assertRaisesRegex(ValueError, 'training resume mismatch: data_sha256'):
                    training(config_path, train_path, out, seed=42, adapter='adapter', resume=ckpt, dev_path=dev_path)
                (ckpt / 'metadata.json').write_text(json.dumps(meta))
                (out / 'dev.jsonl').write_text(json.dumps({'step': 1, 'dataset': 'gsm8k', 'split': 'dev', 'n': 1, 'mean': 1}) + '\n')
                training(config_path, train_path, out, seed=42, adapter='adapter', resume=ckpt, dev_path=dev_path)
                self.assertEqual(json.loads((out / 'run.json').read_text()), original_run)
                self.assertEqual((out / 'env.txt').read_text(), 'original-env\n')

    def _training_fixture(self, root, steps=1):
        import random
        import torch
        from agent_grpo.training import training
        train_path = root / 'train.jsonl'
        dev_path = root / 'dev.jsonl'
        config_path = root / 'config.json'
        ckpt = root / 'checkpoint-0001'
        out = root / 'run'
        ckpt.mkdir()
        out.mkdir()
        write_jsonl(train_path, [Task('tr1', 'gsm8k', 'Q', '2', 'train')])
        write_jsonl(dev_path, [Task('dv1', 'gsm8k', 'Q', '2', 'dev')])
        cfg = {'temperature': 1.0, 'group_size': 2, 'update_epochs': 1, 'steps': steps,
               'learning_rate': 1e-5, 'dtype': 'float32', 'device': 'cpu', 'checkpoint_every': 1,
               'max_new_tokens': 10, 'max_sequence_length': 32, 'questions_per_step': 1,
               'kl_beta': 0.04, 'clip_epsilon': 0.2, 'model': 'fixture'}
        config_path.write_text(json.dumps(cfg))
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        torch.save({'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(), 'step': 1,
                    'sampling_rng': random.Random(42).getstate(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': []}, ckpt / 'training-state.pt')
        meta = {'config': {**cfg, 'seed': 42, 'initial_adapter': 'adapter'}, 'data_sha256': 'same',
                'model_revision_resolved': 'rev1'}
        return training, config_path, train_path, dev_path, ckpt, out, model, meta

    def test_missing_checkpoint_metadata_needs_verifiable_history(self):
        with tempfile.TemporaryDirectory() as directory:
            training, config_path, train_path, dev_path, ckpt, out, model, meta = self._training_fixture(Path(directory))
            (out / 'run.json').write_text('{"wrong_origin": true}\n')
            (out / 'env.txt').write_text('fixture\n')
            with patch('agent_grpo.training.model_and_tokenizer', return_value=(model, None)), \
                 patch('agent_grpo.training.seed_all'), \
                 patch('agent_grpo.training.metadata', side_effect=lambda *args: dict(meta)):
                for info in (None, {}):
                    meta_path = ckpt / 'metadata.json'
                    if info is None and meta_path.exists():
                        meta_path.unlink()
                    if info is not None:
                        meta_path.write_text(json.dumps(info))
                    with self.assertRaisesRegex(ValueError, 'provenance is not verifiable'):
                        training(config_path, train_path, out, seed=42, adapter='adapter', resume=ckpt, dev_path=dev_path)
                self.assertEqual(json.loads((out / 'run.json').read_text()), {'wrong_origin': True})
                history = dict(meta)
                (out / 'run.json').write_text(json.dumps(history))
                meta_path = ckpt / 'metadata.json'
                if meta_path.exists():
                    meta_path.unlink()
                (out / 'dev.jsonl').write_text(json.dumps({'step': 1, 'dataset': 'gsm8k', 'split': 'dev', 'n': 1, 'mean': 1}) + '\n')
                training(config_path, train_path, out, seed=42, adapter='adapter', resume=ckpt, dev_path=dev_path)
                self.assertEqual(json.loads((out / 'run.json').read_text())['data_sha256'], 'same')
                (ckpt / 'metadata.json').write_text(json.dumps({**meta, 'dev_sha256': 'not-the-dev-file'}))
                with self.assertRaisesRegex(ValueError, 'dev_sha256'):
                    training(config_path, train_path, out, seed=42, adapter='adapter', resume=ckpt, dev_path=dev_path)

    def test_resume_backfills_missing_dev_and_does_not_duplicate_logged_steps(self):
        import torch
        from contextlib import nullcontext
        from agent_grpo.schema import Trajectory
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training, config_path, train_path, dev_path, ckpt, out, model, meta = self._training_fixture(root, steps=1)
            model.disable_adapter = lambda: nullcontext()
            (ckpt / 'metadata.json').write_text(json.dumps(meta))
            (out / 'env.txt').write_text('fixture\n')

            def fake_batch(model, tokenizer, tasks, **kwargs):
                return [Trajectory(task.id, [1, 2], [False, True], '<final>2</final>',
                                   [{'assistant': '2', 'generated_tokens': 1, 'tools': []}],
                                   {'total': 1.0, 'correctness': 1.0, 'tool_calls': 0}, False) for task in tasks]

            with patch('agent_grpo.training.model_and_tokenizer', return_value=(model, None)), \
                 patch('agent_grpo.training.seed_all'), \
                 patch('agent_grpo.training.metadata', side_effect=lambda *args: dict(meta)), \
                 patch('agent_grpo.training.batch_tool_trajectories', side_effect=fake_batch) as sampled:
                training(config_path, train_path, out, seed=42, adapter='adapter', resume=ckpt, dev_path=dev_path)
                self.assertEqual(sampled.call_count, 1)
                dev_rows = [json.loads(line) for line in (out / 'dev.jsonl').read_text().splitlines() if line.strip()]
                self.assertEqual([(row['step'], row['dataset']) for row in dev_rows], [(1, 'gsm8k')])
                training(config_path, train_path, out, seed=42, adapter='adapter', resume=ckpt, dev_path=dev_path)
                self.assertEqual(sampled.call_count, 1)
                dev_rows = [json.loads(line) for line in (out / 'dev.jsonl').read_text().splitlines() if line.strip()]
                self.assertEqual(len(dev_rows), 1)

            state = torch.load(ckpt / 'training-state.pt', weights_only=False)
            state['step'] = 1
            torch.save(state, ckpt / 'training-state.pt')
            (out / 'train.jsonl').write_text(json.dumps({'step': 2, 'seed': 42, 'loss': 0.1}) + '\n')
            (out / 'rollouts.jsonl').write_text(json.dumps({'step': 2, 'task_id': 'tr1'}) + '\n')
            config = json.loads(config_path.read_text())
            config['steps'] = 2
            config_path.write_text(json.dumps(config))
            meta['config']['steps'] = 2
            (ckpt / 'metadata.json').write_text(json.dumps(meta))

            def fake_logprobs(model, token_ids, device):
                return next(model.parameters()).reshape(-1)[:1] * 0

            with patch('agent_grpo.training.model_and_tokenizer', return_value=(model, None)), \
                 patch('agent_grpo.training.seed_all'), \
                 patch('agent_grpo.training.metadata', side_effect=lambda *args: dict(meta)), \
                 patch('agent_grpo.training.batch_tool_trajectories', side_effect=fake_batch), \
                 patch('agent_grpo.training.token_logprobs', side_effect=fake_logprobs), \
                 patch('agent_grpo.training.save_checkpoint'):
                training(config_path, train_path, out, seed=42, adapter='adapter', resume=ckpt, dev_path=dev_path)
            train_steps = [json.loads(line)['step'] for line in (out / 'train.jsonl').read_text().splitlines() if line.strip()]
            rollout_steps = [json.loads(line)['step'] for line in (out / 'rollouts.jsonl').read_text().splitlines() if line.strip()]
            self.assertEqual(train_steps, [2])
            self.assertEqual(rollout_steps, [2])

    def test_evaluate_rejects_duplicate_ids_and_foreign_rows_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'test.jsonl'
            config = root / 'config.json'
            out = root / 'eval.jsonl'
            write_jsonl(data, [Task('t1', 'gsm8k', 'Q', '2', 'test'), Task('t2', 'gsm8k', 'Q2', '3', 'test')])
            config.write_text(json.dumps({'temperature': 1.0, 'group_size': 2, 'update_epochs': 1,
                                          'max_new_tokens': 4, 'max_sequence_length': 32, 'model': 'fixture'}))
            row = {'task_id': 't1', 'condition': 'base-direct', 'seed': 42, 'split': 'test'}
            out.write_text(json.dumps(row) + '\n' + json.dumps(row) + '\n')
            (Path(str(out) + '.metadata.json')).write_text(json.dumps({'condition': 'base-direct'}))

            def fail_load(*args, **kwargs):
                raise AssertionError('model loaded before row checks')

            with patch('agent_grpo.stages.model_and_tokenizer', side_effect=fail_load), \
                 patch('agent_grpo.stages.seed_all'):
                with self.assertRaisesRegex(ValueError, 'duplicate evaluation task IDs'):
                    evaluate(config, data, out, 'base-direct')
                self.assertFalse(Path(str(out) + '.executions.jsonl').exists())
                out.write_text(json.dumps({**row, 'condition': 'base-tools'}) + '\n')
                with self.assertRaisesRegex(ValueError, 'condition'):
                    evaluate(config, data, out, 'base-direct')
                out.write_text(json.dumps({**row, 'seed': 43}) + '\n')
                with self.assertRaisesRegex(ValueError, 'seed'):
                    evaluate(config, data, out, 'base-direct')
                out.write_text(json.dumps({**row, 'split': 'dev'}) + '\n')
                with self.assertRaisesRegex(ValueError, 'split'):
                    evaluate(config, data, out, 'base-direct')

    def test_sft_source_and_old_checkpoint_metadata(self):
        import torch
        from agent_grpo.stages import assert_trajectory_source, sft
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'cold.jsonl'
            config = root / 'config.json'
            out = root / 'sft'
            data.write_text(json.dumps({'task_id': 'a', 'token_ids': [1, 2], 'generated_mask': [False, True],
                                        'source_split': 'train', 'reward': {'correctness': 1}}) + '\n')
            cfg = {'temperature': 1.0, 'group_size': 2, 'update_epochs': 1, 'model': 'fixture',
                   'dtype': 'float32', 'device': 'cpu', 'sft_gradient_accumulation': 8}
            config.write_text(json.dumps(cfg))
            Path(str(data) + '.metadata.json').write_text(json.dumps({
                'config': {'model': 'fixture', 'model_revision': 'legacy-rev'}}))
            model = torch.nn.Linear(2, 2)
            model.config = type('Config', (), {'_commit_hash': 'live-rev'})()
            with self.assertRaisesRegex(ValueError, 'another model'):
                assert_trajectory_source(json.loads(Path(str(data) + '.metadata.json').read_text()), cfg, model)
            model.config._commit_hash = 'legacy-rev'
            assert_trajectory_source(json.loads(Path(str(data) + '.metadata.json').read_text()), cfg, model)
            Path(str(data) + '.metadata.json').write_text(json.dumps({
                'config': {'model': 'fixture'}, 'model_revision_resolved': 'legacy-rev'}))

            class Tokenizer:
                def save_pretrained(self, path):
                    Path(path).mkdir(parents=True, exist_ok=True)

            model.save_pretrained = Tokenizer().save_pretrained

            def fake_logprobs(model, token_ids, device):
                return next(model.parameters()).reshape(-1)[:1] * 0

            with patch('agent_grpo.stages.model_and_tokenizer', return_value=(model, Tokenizer())), \
                 patch('agent_grpo.stages.seed_all'), \
                 patch('agent_grpo.stages.token_logprobs', side_effect=fake_logprobs):
                sft(config, data, out, seed=42, epochs=1)
                self.assertEqual([json.loads(line)['step'] for line in (out / 'train.jsonl').read_text().splitlines()], [1])
                checkpoint = out / 'checkpoint-0001'
                state = torch.load(checkpoint / 'training-state.pt', weights_only=False)
                state['step'] = 0
                torch.save(state, checkpoint / 'training-state.pt')
                (checkpoint / 'metadata.json').unlink()
                sft(config, data, out, seed=42, epochs=1, resume=checkpoint)
                self.assertEqual([json.loads(line)['step'] for line in (out / 'train.jsonl').read_text().splitlines()], [1])
                (checkpoint / 'metadata.json').write_text('{}')
                (out / 'run.json').write_text('{"wrong_origin": true}\n')
                with self.assertRaisesRegex(ValueError, 'provenance is not verifiable'):
                    sft(config, data, out, seed=42, epochs=1, resume=checkpoint)


if __name__ == '__main__':
    unittest.main()
