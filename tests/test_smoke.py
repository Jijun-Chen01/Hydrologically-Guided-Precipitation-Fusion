"""Model interface and numerical smoke tests using simulated inputs."""
import unittest
from pathlib import Path
import torch
from src.data import load_model_data
from src.fusion import Rain_HBV, Static_HBV
from src.losses import station_loss
from src.train import train

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'examples/synthetic_data/tiny_demo.npz'


class ModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_model_data(DATA)

    def test_fusion_gradient_and_normalization(self):
        d = self.data
        model = Rain_HBV(24, 4, 9, d.downstream_index, d.routing_order)
        q, _, _, uh, weights, corrected, aet, comp = model(
            d.x_hbv, d.x_rain, d.x_river, d.forcing, d.area, return_comp=True)
        self.assertEqual(tuple(q.shape), (4, 96))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones_like(corrected), atol=1e-5))
        self.assertTrue(torch.allclose(uh.sum(dim=-1), torch.ones(4), atol=1e-5))
        loss, *_ = station_loss(q.index_select(0,d.station_indices), d.qobs_station,
                                torch.isfinite(d.qobs_station), torch.ones(96,dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.rain_net[-1].weight.grad)
        self.assertTrue(torch.isfinite(model.rain_net[-1].weight.grad).all())
        self.assertTrue(torch.isfinite(comp['soil']).all())

    def test_single_baseline(self):
        d = self.data
        model = Static_HBV(24,9,d.downstream_index,d.routing_order,kernel_size=30)
        q, state, params, uh, aet, comp = model(
            d.x_hbv,d.x_river,d.forcing,d.area,return_comp=True)
        self.assertEqual(tuple(q.shape), (4,96))
        self.assertEqual(tuple(params.shape), (4,12))
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue(torch.isfinite(comp['snow']).all())

    def test_one_training_epoch(self):
        config=dict(data_file=str(DATA), output_dir='results/test', model='fusion',
                    seed=2026, epochs=1, patience=1, print_interval=1,
                    periods={'Train':['2007-01-01','2007-03-01'],
                             'Val':['2007-03-01','2007-04-07']},
                    use_process_constraint=True, use_rci_constraint=False,
                    flow_weight=0.8, process_weight=0.2, rci_weight=0,
                    pub_stations=['mock_mid'])
        _, history = train(config, save=False)
        self.assertEqual(len(history),1)


if __name__ == '__main__':
    unittest.main()
