import copy
import random
import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset

class RecoveryEquivalenceTests(unittest.TestCase):
    def test_cpu_interrupted_training(self):
        def setup():
            random.seed(42)
            torch.manual_seed(42)
            model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
            opt = torch.optim.AdamW(model.parameters(), lr=.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=9)
            data = TensorDataset(torch.arange(20, dtype=torch.float32).reshape(10, 2) / 20)
            loader = DataLoader(data, batch_size=3, shuffle=True, generator=torch.Generator().manual_seed(42))
            return model, opt, sched, loader

        def batches(loader):
            while True:
                yield from loader

        def advance(model, opt, sched, iterator, count):
            for _ in range(count):
                x, = next(iterator)
                loss = (model(x) - random.random()).square().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()

        model, opt, sched, loader = setup()
        advance(model, opt, sched, batches(loader), 9)
        expected = copy.deepcopy(model.state_dict())
        expected_scheduler = copy.deepcopy(sched.state_dict())
        model, opt, sched, loader = setup()
        iterator = batches(loader)
        advance(model, opt, sched, iterator, 5)
        checkpoint = copy.deepcopy((model.state_dict(), opt.state_dict(), sched.state_dict(), random.getstate(), torch.get_rng_state()))
        model, opt, sched, loader = setup()
        model.load_state_dict(checkpoint[0])
        opt.load_state_dict(checkpoint[1])
        sched.load_state_dict(checkpoint[2])
        iterator = batches(loader)
        for _ in range(5):
            next(iterator)
        random.setstate(checkpoint[3])
        torch.set_rng_state(checkpoint[4])
        advance(model, opt, sched, iterator, 4)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, expected[key]), key)
        self.assertEqual(sched.state_dict(), expected_scheduler)

if __name__ == "__main__":
    unittest.main()
