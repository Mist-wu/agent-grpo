"""Real tiny Qwen+PEFT CPU checks. No downloaded weights or benchmark claims."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

@unittest.skipUnless(importlib.util.find_spec('transformers') and importlib.util.find_spec('peft'), 'install train extra')
class TinyQwenTests(unittest.TestCase):
    def test_real_qwen_lora_gradient_and_checkpoint_roundtrip(self):
        import torch
        import random
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from peft import LoraConfig, get_peft_model
        from agent_grpo.training import token_logprobs, save_checkpoint
        from agent_grpo.objective import grpo_loss
        torch.manual_seed(7)
        model=get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,
              num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64)),
              LoraConfig(r=2,lora_alpha=4,lora_dropout=0,target_modules='all-linear',task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        tokens=[1,3,5,7,9,2]
        with torch.no_grad():
            old=token_logprobs(model,tokens,'cpu')
            with model.disable_adapter(): ref=token_logprobs(model,tokens,'cpu')
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.01)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        loss,_=grpo_loss(token_logprobs(model,tokens,'cpu'),old,ref,torch.tensor([False,True,False,True,True]),1)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad))
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            class Tokenizer:
                def save_pretrained(self,path): (Path(path)/'tokenizer.stub').write_text('test only')
            path=Path(directory)/'checkpoint-0001'
            save_checkpoint(path,model,Tokenizer(),optimizer,scaler,1,random.Random(42),{'test_only':True})
            self.assertTrue((path/'adapter_model.safetensors').exists())
            state=torch.load(path/'training-state.pt',weights_only=False)
            self.assertEqual(state['step'],1)
            optimizer.load_state_dict(state['optimizer'])
            self.assertFalse(path.with_name(path.name+'.partial').exists())

if __name__=='__main__':unittest.main()
