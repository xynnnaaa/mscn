import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from mscn.mixture import SingleSampleMixture, load_mixture_samples


class QueryOnlyMixtureTests(unittest.TestCase):
    def test_loader_needs_only_four_files_and_drops_log_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = dict(use_adaptive_single_mixture=True, use_single_embedding=1,
                     single_bitmap_dim=8, single_embedding_dim=3, single_pca_dim=2)
            for source in ('query_aware', 'random'):
                bitmap = root / (source + '.bitmaps')
                bitmap.write_bytes((1).to_bytes(4, 'little') + b'\x00')
                embedding = root / (source + '.pt')
                torch.save({0: {'t': torch.tensor([1., 2., 4., 99., 0., 0.])}}, embedding)
                c['train_' + source + '_bitmap_file'] = str(bitmap)
                c['train_' + source + '_embedding_file'] = str(embedding)
            # A stale metadata key must not be opened or required.
            c['train_single_mixture_metadata_file'] = '/nonexistent/metadata.json'
            row = load_mixture_samples(c, 'train', [['table t']], {'table t': [1.]})[0][0]
            self.assertEqual(row.shape, (27,))
            for offset in (1, 14):
                np.testing.assert_array_equal(row[offset + 8:offset + 11], [1, 2, 4])
                np.testing.assert_array_equal(row[offset + 11:offset + 13], 0)
            self.assertNotIn(99, row)

    def test_alpha_independent_of_sample_values_and_hits(self):
        for mode in ('learned', 'fixed'):
            model = SingleSampleMixture(1, 2, bitmap_dim=2, embedding_dim=3, pca_dim=2,
                                        gate_mode=mode, fixed_alpha=.25)
            if mode == 'learned':
                with torch.no_grad():
                    model.gate[-1].weight.fill_(.2)
                self.assertEqual(model.gate[0].in_features, 5)
            features = torch.randn(1, 5, model.input_dim)
            # Same query context, varied sample evidence including both empty.
            features[0, :, :2] = torch.tensor([[1,0],[1,0],[0,0],[0,0],[0,0]])
            features[0, :, 7:9] = torch.tensor([[1,0],[0,0],[1,0],[0,0],[0,0]])
            mask = torch.tensor([[[1.],[1.],[1.],[1.],[0.]]])
            out = model(features, torch.ones(1,5,1), torch.ones(1,2), torch.ones(1,2), mask)
            alpha = model.last_alpha.flatten()
            torch.testing.assert_close(alpha[:4], alpha[0].expand(4))
            self.assertEqual(alpha[-1].item(), 0)
            self.assertEqual(out[0,-1].abs().sum().item(), 0)
            torch.testing.assert_close(model.last_stats, torch.tensor([4.,1.,alpha[0],1.,1.,1.]))

    def test_dynamic_pca_mask_after_bias_and_no_pca(self):
        for dim in (0,2,3):
            model=SingleSampleMixture(1,2,bitmap_dim=2,embedding_dim=3,pca_dim=dim)
            source=torch.zeros(1,1,model.source_dim)
            source[...,2:5]=torch.tensor([1.,2.,4.])
            if dim:
                with torch.no_grad():
                    model.pca_norm.bias.fill_(3)
                    if hasattr(model,'pca_proj'):
                        model.pca_proj.weight.zero_(); model.pca_proj.bias.fill_(5)
            out=model._encode(source,torch.ones(1,1,1))
            if dim:
                self.assertEqual(out[...,131:].abs().sum().item(),0)
                source[...,5]=1
                self.assertGreater(model._encode(source,torch.ones(1,1,1))[...,131:].abs().sum().item(),0)
            else:
                self.assertEqual(out.shape[-1],131)

    def test_empty_branch_and_gate_receive_gradients(self):
        model=SingleSampleMixture(1,2,bitmap_dim=2,embedding_dim=3,pca_dim=0)
        features=torch.zeros(1,1,10)
        features[...,2:5]=torch.tensor([1.,2.,4.])
        features[...,7:10]=torch.tensor([4.,1.,2.])
        features[...,5]=1  # only random bitmap hits
        features.requires_grad_()
        model(features,torch.ones(1,1,1),torch.zeros(1,2),torch.zeros(1,2),torch.ones(1,1,1))[...,128].sum().backward()
        self.assertGreater(features.grad[...,2:5].abs().sum().item(),0)
        self.assertGreater(features.grad[...,7:10].abs().sum().item(),0)
        self.assertGreater(model.gate[-1].bias.grad.abs().sum().item(),0)


if __name__ == '__main__': unittest.main()
