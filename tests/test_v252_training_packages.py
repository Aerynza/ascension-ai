import json
import unittest
from scripts.validate_v252_packages import ROOT, validate

class PackageTests(unittest.TestCase):
    def setUp(self):
        self.train=[json.loads(s) for s in (ROOT/'evals/training/aerynza_v252_execution_train.jsonl').read_text().splitlines()]
        self.benchmark=json.loads((ROOT/'evals/training/aerynza_v252_execution_benchmark.json').read_text())
        self.prior=[json.loads(s) for s in (ROOT/'evals/training/aerynza_v251_foundation_heldout.jsonl').read_text().splitlines()]
    def test_valid(self):
        self.assertFalse(validate(self.train,self.benchmark,self.prior)['frontier_parity_established'])
    def test_overlap(self):
        self.train[0]['user']=self.prior[0]['user']
        with self.assertRaisesRegex(ValueError,'overlap'): validate(self.train,self.benchmark,self.prior)
    def test_benchmark_leak(self):
        self.benchmark['cases'][0]['task']=self.train[0]['user']
        with self.assertRaisesRegex(ValueError,'overlap'): validate(self.train,self.benchmark,self.prior)
    def test_private_data(self):
        self.train[0]['source_kind']='real_user_conversation'
        with self.assertRaisesRegex(ValueError,'provenance'): validate(self.train,self.benchmark,self.prior)
    def test_false_execution(self):
        self.benchmark['cases'][0]['result']='passed'
        with self.assertRaisesRegex(ValueError,'execution'): validate(self.train,self.benchmark,self.prior)
    def test_duplicate(self):
        self.train[1]['id']=self.train[0]['id']
        with self.assertRaisesRegex(ValueError,'duplicate'): validate(self.train,self.benchmark,self.prior)

if __name__=='__main__': unittest.main()
