"""Checkpoint sync uses immutable sources and fresh verified receipts, never shell."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from agent_grpo.persistence import (CheckpointPersistenceError, parse_sync_hook,
    require_checkpoint_sync, write_checkpoint_manifest, verify_checkpoint_files, persist_checkpoint)

RECEIPT = """import json,sys
from pathlib import Path
source,run_id,receipt,experiment=sys.argv[1:]
assert Path(source,'checkpoint-manifest.json').exists()
assert not Path(source+'.partial').exists()
json.dump(dict(state='verified',source=source,run_id=run_id,experiment=experiment,destination='gdrive:exp/runs/'+experiment+'/'+run_id,verified_at='test-time'),open(receipt,'w'))
"""

class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.env=patch.dict(os.environ,{},clear=True)
        self.env.start()
    def tearDown(self):
        self.env.stop();self.tmp.cleanup()
    def checkpoint(self,name):
        path=self.root/name;path.mkdir()
        (path/'adapter.safetensors').write_bytes(b'test weights')
        (path/'metadata.json').write_text('{}')
        write_checkpoint_manifest(path)
        return path
    def hook(self,code=RECEIPT):
        os.environ['EXP_CHECKPOINT_SYNC']=json.dumps([sys.executable,'-c',code,'{source}','{run_id}','{receipt}','{experiment}'])
    def test_optional_smoke_and_required_production(self):
        path=self.checkpoint('checkpoint-0001')
        self.assertEqual(persist_checkpoint(path)['state'],'local_only')
        self.assertFalse((self.root/'durable-checkpoint.json').exists())
        require_checkpoint_sync(self.root,formal=False)
        with self.assertRaises(CheckpointPersistenceError):require_checkpoint_sync(self.root,formal=True)
        with self.assertRaises(CheckpointPersistenceError):require_checkpoint_sync('/content/test',formal=False)
    def test_validated_json_argv(self):
        for value in ['echo hello','"command"','[]','[1]','["echo","{source}"]']:
            with self.assertRaises(ValueError):parse_sync_hook(value)
        self.hook();self.assertEqual(parse_sync_hook()[0],sys.executable)
    def test_verified_fresh_receipt_and_exact_source(self):
        path=self.checkpoint('checkpoint-0001');self.hook()
        state=persist_checkpoint(path)
        self.assertEqual(state['state'],'verified')
        self.assertEqual(json.loads((self.root/'durable-checkpoint.json').read_text()),state)
        self.assertEqual(json.loads(Path(state['receipt']).read_text())['source'],str(path.resolve()))
        # A no-op success cannot reuse the old valid receipt.
        self.hook('pass')
        with self.assertRaises(CheckpointPersistenceError):persist_checkpoint(path)
        self.assertEqual(json.loads((self.root/'durable-checkpoint.json').read_text()),state)
        self.assertTrue(Path(state['receipt']).exists())
        self.assertEqual(json.loads((self.root/'checkpoint-status.json').read_text())['state'],'sync_failed')
    def test_failure_keeps_prior_durable_and_new_local(self):
        first=self.checkpoint('checkpoint-0001');self.hook();state=persist_checkpoint(first)
        second=self.checkpoint('checkpoint-0002');self.hook('import sys;sys.exit(4)')
        with self.assertRaises(CheckpointPersistenceError):persist_checkpoint(second)
        self.assertTrue(second.exists());self.assertTrue(first.exists())
        self.assertEqual(json.loads((self.root/'durable-checkpoint.json').read_text()),state)
    def test_mutation_during_sync_rejected(self):
        path=self.checkpoint('checkpoint-0001')
        self.hook(RECEIPT+"\nPath(source,'adapter.safetensors').write_bytes(b'changed')\n")
        with self.assertRaises(CheckpointPersistenceError):persist_checkpoint(path)
        self.assertFalse((self.root/'durable-checkpoint.json').exists())
    def test_timeout_keeps_local_checkpoint(self):
        path=self.checkpoint('checkpoint-0001');self.hook('import time;time.sleep(30)')
        with self.assertRaises(CheckpointPersistenceError):persist_checkpoint(path,timeout=.1)
        self.assertTrue(path.exists())
        self.assertFalse((self.root/'durable-checkpoint.json').exists())
    def test_restore_checks_hashes_and_unsafe_inventory(self):
        path=self.checkpoint('checkpoint-0001');verify_checkpoint_files(path)
        (path/'adapter.safetensors').write_bytes(b'truncated')
        with self.assertRaises(ValueError):verify_checkpoint_files(path)

if __name__=='__main__':unittest.main()
