"""Stage recovery and truthful completion gates with controlled model outputs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from agent_grpo.schema import Task, Trajectory, write_jsonl
from agent_grpo.stages import collect


class ResumeTests(unittest.TestCase):
    def test_collection_resume_skips_accepted_and_validates_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data=root/'train.jsonl';config=root/'config.json';out=root/'cold.jsonl'
            write_jsonl(data,[Task('a','gsm8k','Q','2','train'),Task('b','gsm8k','Q2','3','train')])
            config.write_text(json.dumps({'temperature':1.0,'group_size':2,'update_epochs':2,'max_new_tokens':10,'max_sequence_length':30,'model':'fixture'}))
            meta={'config':json.loads(config.read_text()),'data_sha256':'abc','model_revision_resolved':'fixture'}
            trajectory=Trajectory('a',[1,2],[False,True],'<final>2</final>',[],{'correctness':1})
            def sample(model,tokenizer,task,**kwargs):
                return Trajectory(task.id,[1,2],[False,True],'<final>2</final>',[],{'correctness':1})
            with patch('agent_grpo.stages.model_and_tokenizer',return_value=(None,None)),patch('agent_grpo.stages.seed_all'),patch('agent_grpo.stages.metadata',side_effect=lambda *args:dict(meta)),patch('agent_grpo.stages.sample_trajectory',side_effect=sample) as sampling:
                collect(config,data,out,target=1)
                self.assertEqual(sampling.call_count,1)
                collect(config,data,out,target=1)
                self.assertEqual(sampling.call_count,1)
                record=json.loads(Path(str(out)+'.metadata.json').read_text())
                self.assertEqual(record['accepted'],1)
                self.assertEqual(record['resume_count'],1)
                meta['data_sha256']='changed'
                with self.assertRaisesRegex(ValueError,'data_sha256'):
                    collect(config,data,out,target=1)

if __name__=='__main__':unittest.main()
