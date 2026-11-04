#!/usr/bin/env python3
"""Bounded real 0.5B rollout smoke, not an accuracy/performance benchmark."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agent_grpo.schema import Task
from agent_grpo.training import model_and_tokenizer,seed_all
from agent_grpo.rollout import sample_trajectory
import json
import time
import torch

torch.set_num_threads(1)
config={'model':'Qwen/Qwen2.5-0.5B-Instruct','model_revision':'7ae557604adf67be50417f59c2c2f167def9a775','dtype':'float32','device':'cpu'}
seed_all(42)
model,tokenizer=model_and_tokenizer(config)
tasks=[Task('smoke:1','gsm8k','What is 2 plus 3?','5','dev'),
       Task('smoke:2','hotpotqa','In what city was Alice born?','Tokyo','dev',[{'title':'Alice','text':'Alice was born in Tokyo.'}])]
rows=[]
for task in tasks:
    started=time.monotonic()
    trajectory=sample_trajectory(model,tokenizer,task,temperature=0,max_new_tokens=32,max_sequence_length=512,max_tool_calls=2)
    assert len(trajectory.token_ids)==len(trajectory.generated_mask) and any(trajectory.generated_mask)
    rows.append({'id':task.id,'seconds':time.monotonic()-started,'tokens':len(trajectory.token_ids),
                 'generated_tokens':sum(trajectory.generated_mask),'turns':trajectory.turns,
                 'reward':trajectory.reward,'truncated':trajectory.truncated})
result={'smoke':True,'benchmark':False,'model':config,'torch':torch.__version__,'results':rows}
out=Path(__file__).resolve().parents[1]/'results/smoke-cpu.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
