#!/usr/bin/env python3
"""Download tokenizer only and prove inserted tool suffix matches native Qwen chat."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from transformers import AutoTokenizer
from huggingface_hub import HfApi
from agent_grpo.rollout import initial_prompt, tool_suffix, SYSTEM
from agent_grpo.schema import Task
from agent_grpo.tools import tool_schemas
from agent_grpo.tool_format import render_tool_calls
import json

model='Qwen/Qwen2.5-0.5B-Instruct'
revision=HfApi().model_info(model).sha
tokenizer=AutoTokenizer.from_pretrained(model,revision=revision)
task=Task('probe','gsm8k','What is 1 + 1?','2','train')
call={'name':'calculator','arguments':{'expr':'1+1'}}
response={'ok':True,'result':'2'}
prefix=initial_prompt(tokenizer,task,True)
assistant=render_tool_calls([call])
# Template formatting may serialize JSON spaces differently; use text assistant to preserve sampled bytes.
generated=tokenizer.encode(assistant,add_special_tokens=False)
actual=prefix+generated+tool_suffix(tokenizer,response,generated)
messages=[{'role':'system','content':SYSTEM},{'role':'user','content':task.question},
          {'role':'assistant','content':assistant},{'role':'tool','content':json.dumps(response,ensure_ascii=False)}]
expected=tokenizer.apply_chat_template(messages,tools=tool_schemas(),tokenize=True,add_generation_prompt=True)
if tokenizer.decode(actual) != tokenizer.decode(expected):
    import difflib
    print('\n'.join(difflib.unified_diff(tokenizer.decode(expected).splitlines(),tokenizer.decode(actual).splitlines())))
    raise SystemExit('tool suffix is incompatible with native chat template')
print(json.dumps({'passed':True,'model':model,'revision':revision,'test':'native tool-response suffix'},indent=2))
