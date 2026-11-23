import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from agent_grpo.data import normalize_row, split_train
from agent_grpo.objective import group_advantages, scalar_loss
from agent_grpo.rewards import SCORE_VERSION, hotpot_f1, numeric_answer, reward
from agent_grpo.rollout import _cut_at_eos, _finish_turn, _new_state, append_segment
from agent_grpo.schema import Task, Trajectory, write_jsonl
from agent_grpo.tools import calculator, search, execute, probe_python_sandbox, python_sandbox
from agent_grpo.tool_format import parse_tool_calls
from agent_grpo.report import rescore_saved, summarize


class CoreTests(unittest.TestCase):
    def test_calculator_valid_and_exploit_rejection(self):
        self.assertEqual(calculator('-(3+4)*2/7'), '-2.0')
        for expr in ["__import__('os').system('id')", '2**999999', '(1).__class__', '[1][0]', 'True+2', '1e309', '1/0']:
            with self.assertRaises((ValueError, ZeroDivisionError)):
                calculator(expr)

    def test_strict_native_parser(self):
        self.assertEqual(parse_tool_calls('hello'), [])
        self.assertEqual(parse_tool_calls('<tool_call>{"name":"calculator","arguments":{"expr":"1+1"}}</tool_call>')[0]['name'], 'calculator')
        for text in ['<tool_call>{}</tool_call>', '<tool_call>', '</tool_call>', '<tool_call>{"name":"x","name":"y","arguments":{}}</tool_call>', '<tool_call>{"name":"x","arguments":{"x":NaN}}</tool_call>']:
            with self.assertRaises(ValueError):
                parse_tool_calls(text)

    def test_local_bm25(self):
        docs=[{'title':'A','text':'Alice was born in Tokyo.'}, {'title':'B','text':'Bob lives in Paris.'}]
        self.assertEqual(search('Alice Tokyo', docs)[0]['title'], 'A')
        self.assertEqual(search('unfindable',docs), [])
        self.assertFalse(execute({'name':'python','arguments':{'code':'print(1)'}},[])['ok'])

    def test_python_never_falls_back_to_host(self):
        with patch('agent_grpo.tools.shutil.which', return_value=None):
            with self.assertRaises(RuntimeError):
                python_sandbox("open('/tmp/forbidden','w')")
            self.assertFalse(probe_python_sandbox())

    def test_mask_preserves_generated_tokens_and_excludes_tool_results(self):
        ids, mask=[1,2],[False,False]
        append_segment(ids,mask,[3,4],True)
        append_segment(ids,mask,[5,6],False)
        append_segment(ids,mask,[7],True)
        self.assertEqual([i for i,m in zip(ids,mask) if m],[3,4,7])
        Trajectory('test',ids,mask,'',[],{})
        with self.assertRaises(ValueError):
            Trajectory('test',[1],[True],'',[],{})

    def test_group_normalization_and_all_equal(self):
        self.assertEqual(group_advantages([1,1,1]),[0,0,0])
        values=group_advantages([0,1])
        self.assertAlmostEqual(sum(values),0)
        self.assertAlmostEqual(values[1],1,places=5)
        with self.assertRaises(ValueError):
            group_advantages([1])

    def test_clipped_loss_signs_and_kl(self):
        # ratio 2, positive advantage -> 1.2; negative advantage -> -2.
        self.assertAlmostEqual(scalar_loss([math.log(.4)],[math.log(.2)],[math.log(.4)],[True],1),-1.2)
        self.assertAlmostEqual(scalar_loss([math.log(.4)],[math.log(.2)],[math.log(.4)],[True],-1),2)
        self.assertGreater(scalar_loss([-.5],[-.5],[-2],[True],0),0)
        # Masked tool token can't change loss even with enormous fake log-prob.
        self.assertAlmostEqual(scalar_loss([0,-1000],[0,-1000],[0,1000],[True,False],1),-1)

    def test_answer_metrics_do_not_guess_from_working(self):
        self.assertEqual(numeric_answer('reasoning\n#### 1,234'),1234)
        self.assertIsNone(numeric_answer('I computed 3 and then 4.'))
        self.assertEqual(numeric_answer('The result of 2 plus 3 is 5.'),5)
        self.assertEqual(numeric_answer(r'We get \boxed{42}'),42)
        self.assertEqual(hotpot_f1('<final>The New York</final>','New York'),1)
        self.assertEqual(hotpot_f1('yes','yes indeed'),0)
        task=Task('x','gsm8k','Q','#### 2','train')
        r=reward(task,'<final>2</final>',4,4)
        self.assertAlmostEqual(r['total'],.9)  # one parsing bonus, not four

    def test_source_adapters_and_disjoint_splits(self):
        row={'id':'x','question':'Q','answer':'yes','context':{'title':['A'],'sentences':[['One.','Two.']]}}
        task=normalize_row('hotpotqa',row,'validation',0,'abc')
        self.assertEqual(task.split,'test')
        self.assertEqual(task.documents[0]['text'],'One. Two.')
        tasks=[Task(str(i),'gsm8k',f'Q{i}','1','train') for i in range(20)]
        train,dev=split_train(tasks,10,5,42)
        self.assertFalse({t.id for t in train}&{t.id for t in dev})
        self.assertTrue(all(t.split=='dev' for t in dev))
        self.assertTrue(all(t.split=='train' for t in tasks))
        symbolic=normalize_row('gsm-symbolic',{'id':4,'instance':2,'question':'Q','answer':'#### 1','original_id':8},'test',0)
        self.assertEqual(symbolic.source['template_id'],4)
        self.assertEqual(symbolic.id,'gsm-symbolic:test:4:2')

    def test_report_clusters_symbolic(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'eval.jsonl'
            meta={'split':'test','tasks':4}
            Path(str(path)+'.metadata.json').write_text(json.dumps(meta))
            rows=[{'task_id':str(i),'condition':'grpo','seed':42,'dataset':'gsm-symbolic','split':'test',
                   'source':{'template_id':i//2},
                   'reward':{'correctness':float(i<2),'strict_correctness_v1':float(i<2),
                             'final_format_compliant':True,'tool_calls':1,'score_version':SCORE_VERSION},
                   'truncated':False} for i in range(4)]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            summary=summarize([path],Path(directory)/'out.json')
            self.assertEqual(summary['evaluations'][0]['mean'],.5)
            self.assertEqual(summary['evaluations'][0]['ci_unit'],'template')
            with self.assertRaisesRegex(ValueError,'duplicate task ID across inputs'):
                summarize([path,path],Path(directory)/'dup.json')

    def test_rescore_preserves_original_reward_across_reruns(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data=root/'test.jsonl'
            write_jsonl(data,[Task('t1','gsm8k','Q','#### 42','test')])
            raw=root/'raw.jsonl'
            orig={'total':0.05,'correctness':0.0,'parse_bonus':0.1,'tool_penalty':0.05,'tool_calls':1}
            raw.write_text(json.dumps({'task_id':'t1','final_text':'Answer is 42','reward':orig})+'\n')
            Path(str(raw)+'.metadata.json').write_text(json.dumps({'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'tasks':1}))
            v2=root/'v2.jsonl'
            v2b=root/'v2b.jsonl'
            rescore_saved(data,raw,v2)
            rescore_saved(data,v2,v2b)
            row=json.loads(v2b.read_text().strip())
            self.assertEqual(row['original_reward'],orig)
            self.assertEqual(row['reward']['correctness'],1.0)
            self.assertAlmostEqual(row['reward']['total'],1.05)

    def test_eos_cut_drops_only_tokens_after_first_eos(self):
        self.assertEqual(_cut_at_eos([4, 5, 9, 9], 9), [4, 5, 9])
        self.assertEqual(_cut_at_eos([9], 9), [9])
        self.assertEqual(_cut_at_eos([4, 5], 9), [4, 5])

    def test_truncated_malformed_tool_call_marks_truncated(self):
        class FakeTokenizer:
            eos_token_id=99
            def decode(self,ids,skip_special_tokens=True):
                return '<tool_call>{"name": "calculator", "arguments": {"expr": "1+'
        state=_new_state([1,2])
        task=Task('t1','gsm8k','Q','2','test')
        _finish_turn(state,task,FakeTokenizer(),[10,11,12,13],turn_budget=4,
                     tools_enabled=True,max_tool_calls=4,max_sequence_length=32,enable_python=False)
        self.assertTrue(state['truncated'])
        self.assertFalse(state['active'])
        self.assertIn('parse_error',state['turns'][0])


if __name__=='__main__': unittest.main()
