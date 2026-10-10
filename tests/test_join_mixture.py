import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from mscn.mixture import JoinSampleMixture, join_mixture_options, load_join_mixture_samples
from mscn.model import SetConv
from mscn.data import load_and_encode_all_data


class JoinMixtureTests(unittest.TestCase):
    def test_mask_after_norm_and_unprojected_bitmap(self):
        m = JoinSampleMixture(2, 4, bitmap_dim=2, embedding_dim=3, pca_dim=3)
        self.assertFalse(hasattr(m, 'bitmap_proj'))
        with torch.no_grad():
            m.pca_norm.bias.fill_(7)
        x = torch.tensor([[1., 0., 1., 2., 4., 0., 0., 0.],
                          [0., 1., 1., 2., 4., 1., 2., 3.]])
        y = m._encode(x)
        torch.testing.assert_close(y[:, :2], x[:, :2])
        self.assertEqual(y[0, 5:].abs().sum(), 0)
        self.assertGreater(y[1, 5:].abs().sum(), 0)
        self.assertGreater(y[0, 2:5].abs().sum(), 0)

    def test_gate_query_only_table_padding_and_fixed_fusion(self):
        m = JoinSampleMixture(2, 4, bitmap_dim=2, embedding_dim=3, pca_dim=0)
        x = torch.randn(2, 10)
        tables = torch.tensor([[[1.,0.],[0.,1.],[1.,0.]]]*2)
        mask = torch.tensor([[[1.],[1.],[0.]]]*2)
        ctx = torch.ones(2, 4)
        with torch.no_grad(): m.gate[-1].weight.fill_(.2)
        m(x, tables, ctx, ctx, mask);alpha=m.last_alpha.clone()
        tables[:, 2] = 100
        m(x * 7, tables, ctx, ctx, mask)
        torch.testing.assert_close(alpha, m.last_alpha)
        m.gate_mode='fixed';m.fixed_alpha=.25
        expected=.25*m._encode(x[:,:5])+.75*m._encode(x[:,5:])
        torch.testing.assert_close(m(x,tables,ctx,ctx,mask),expected)
        m.gate_mode='learned'
        with torch.no_grad(): m.gate[-1].weight.zero_();m.gate[-1].bias.zero_()
        m(x,tables,ctx,ctx,mask)
        torch.testing.assert_close(m.last_alpha,torch.full((2,1),.5))

    def test_loader_validation_and_config(self):
        self.assertIsNone(join_mixture_options({}))
        with self.assertRaises(ValueError):join_mixture_options(dict(use_adaptive_join_mixture=True,use_join_embedding=1))
        with tempfile.TemporaryDirectory() as d:
            c=dict(use_adaptive_join_mixture=True,use_join_embedding=2,
                   join_bitmap_dim=2,join_embedding_dim=3,join_pca_dim=0)
            for source in ['query_aware','random']:
                p=Path(d)/(source+'.pt');torch.save({'0':torch.arange(5.),'1':torch.ones(5)},p)
                c[f'train_join_{source}_embedding_file']=str(p)
            x=load_join_mixture_samples(c,'train',2)
            self.assertEqual(x.shape,(2,10));np.testing.assert_array_equal(x[0,:5],np.arange(5))
            p=Path(c['train_join_random_embedding_file'])
            for bad in [{0:torch.zeros(6),1:torch.ones(5)}, {0:torch.full((5,),float('nan')),1:torch.ones(5)}, {0:torch.ones(5),2:torch.ones(5)}]:
                torch.save(bad,p)
                with self.assertRaises(ValueError):load_join_mixture_samples(c,'train',2)

    def test_model_combinations_backward_and_checkpoint(self):
        opts=dict(bitmap_dim=2,embedding_dim=3,pca_dim=3,gate_hidden=4)
        for single,join in [(False,False),(False,True),(True,False),(True,True)]:
            with self.subTest(single=single,join=join):
                model=SetConv(2,16 if single else 2,4,2,16 if join else 0,8,
                              use_single_embedding=int(single),dropout_p=0,
                              single_mixture_options=opts if single else None,
                              join_mixture_options=opts if join else None)
                samples=torch.randn(3,2,18 if single else 4);samples[:,:,:2]=torch.eye(2)
                args=(samples,torch.randn(3,2,4),torch.randn(3,1,2),torch.randn(3,16 if join else 1),
                      torch.ones(3,2,1),torch.ones(3,2,1),torch.ones(3,1,1))
                y=model(*args);self.assertEqual(y.shape,(3,1));self.assertTrue(torch.isfinite(y).all())
                y.sum().backward()
                if join:self.assertIsNotNone(model.join_mixture.gate[-1].weight.grad)
                clone=copy.deepcopy(model);clone.load_state_dict(model.state_dict())
                torch.testing.assert_close(y,clone(*args))

    def test_data_pipeline_train_val_split(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'train.csv').write_text(''.join(f'a a##a.x,>,0#{i+1}\n' for i in range(10)))
            (p/'test.csv').write_text('a a##a.x,>,0#2\na a##a.x,>,0#3\n')
            (p/'cols.csv').write_text('name,min,max\na.x,0,10\n')
            c=dict(workloads_dir=d,trainset='train',testset='test',column_min_max_file=str(p/'cols.csv'),
                   use_single_embedding=-1,use_join_embedding=2,use_adaptive_join_mixture=True,
                   join_bitmap_dim=2,join_embedding_dim=3,join_pca_dim=0)
            for split,n in [('train',10),('test',2)]:
                for source in ['query_aware','random']:
                    path=p/f'{split}_{source}.pt';torch.save({i:torch.tensor([i,0,1,2,4.]) for i in range(n)},path)
                    c[f'{split}_join_{source}_embedding_file']=str(path)
            data=load_and_encode_all_data(c)
            train,val,test=data[4:7]
            self.assertEqual((len(train),len(val),len(test)),(9,1,2))
            self.assertEqual(train[0][3].shape,(10,))
            self.assertEqual(val[0][3][0],9)
            self.assertEqual(test[1][3][0],1)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
