import importlib.util
import math
import unittest

@unittest.skipUnless(importlib.util.find_spec('torch'), 'install train extra for gradient tests')
class TorchTests(unittest.TestCase):
    def test_objective_matches_oracle_and_masks_gradients(self):
        import torch
        from agent_grpo.objective import grpo_loss, scalar_loss
        current=torch.tensor([-.3,-.4,-.5],requires_grad=True)
        old=torch.tensor([-.35,-.45,-.55],requires_grad=True)
        reference=torch.tensor([-.5,-.7,-.9],requires_grad=True)
        mask=torch.tensor([True,False,True])
        loss,_=grpo_loss(current,old,reference,mask,.7)
        oracle=scalar_loss(current.detach().tolist(),old.detach().tolist(),reference.detach().tolist(),mask.tolist(),.7)
        self.assertAlmostEqual(float(loss.detach()),oracle,places=6)
        loss.backward()
        self.assertEqual(float(current.grad[1]),0)
        self.assertIsNone(old.grad)
        self.assertIsNone(reference.grad)

    def test_zero_advantage_at_reference_has_zero_gradient(self):
        import torch
        from agent_grpo.objective import grpo_loss
        current=torch.tensor([-.3,-.4],requires_grad=True)
        loss,_=grpo_loss(current,current.detach(),current.detach(),torch.tensor([True,True]),0)
        loss.backward()
        self.assertTrue(torch.equal(current.grad,torch.zeros(2)))

    def test_positive_clipped_policy_gradient_is_zero(self):
        import torch
        from agent_grpo.objective import grpo_loss
        current=torch.tensor([math.log(.4)],requires_grad=True)
        loss,_=grpo_loss(current,torch.tensor([math.log(.2)]),current.detach(),torch.tensor([True]),1,beta=0)
        loss.backward()
        self.assertEqual(float(current.grad[0]),0)

    def test_selective_logprobs_match_cross_entropy_and_backward(self):
        import torch
        from agent_grpo.training import token_logprobs
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.table=torch.nn.Parameter(torch.randn(7,11))
            def forward(self,ids,use_cache=False):
                return type('Output',(),{'logits':self.table[ids]})()
        model=Model();tokens=[1,3,4,2]
        selected=token_logprobs(model,tokens,'cpu')
        expected=torch.log_softmax(model.table[tokens[:-1]],dim=-1).gather(1,torch.tensor(tokens[1:])[:,None]).squeeze(1)
        torch.testing.assert_close(selected,expected)
        selected.sum().backward()
        self.assertTrue(torch.isfinite(model.table.grad).all())

if __name__=='__main__':unittest.main()
