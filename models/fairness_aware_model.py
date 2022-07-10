# @Filename:    fairness_aware_model.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/8/22 12:57 PM

import gensim
import gravlearn
from tqdm import tqdm
import torch
from models.model import Model
from utils.config_utils import get_sk_value, CONSTANTS

class FairnessAwareModel(Model):

    def __init__(self, device, num_nodes, dim):
        super().__init__(device, num_nodes, dim)
        self.model = gravlearn.Word2Vec(vocab_size=num_nodes, dim=dim)
        self.device = device

    def save(self, path, biased_wv=None):
        if biased_wv is None:
            # TODO(ashutiwa): load biased wv from path
            pass
        in_vec = self.model.ivectors.weight.detach().cpu().numpy()
        kv = gensim.models.KeyedVectors(in_vec.shape[1])
        kv.add_vectors(biased_wv.index_to_key, in_vec)
        kv.save(path)

    def fit(self, dataset: gravlearn.DataLoader, workers=4, iid=None, ):
        assert not iid, 'iid not yet supported in FairnessAwareModel'
        self.device = next(self.model.parameters()).device
        self.model.train()
        self.model = self.model.to(self.device)
        self.device = next(self.model.parameters()).device
        loss_func = gravlearn.TripletLoss(embedding=self.model, dist_metric=get_sk_value(CONSTANTS.DIST_METRIC.__name__.lower(), CONSTANTS.PARAMS, object=True))
        checkpoint = get_sk_value("checkpoint", CONSTANTS.PARAMS)
        focal_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optim = torch.optim.AdamW(focal_params, lr=get_sk_value("lr", CONSTANTS.PARAMS))
        pbar = tqdm(enumerate(dataset), total=len(dataset), miniters=100)
        for it, (p1, p2, n1) in pbar:
            focal_params = filter(lambda p: p.requires_grad, self.model.parameters())
            for param in focal_params:
                param.grad = None

            # convert to bags if bags are given
            p1, p2, n1 = p1.to(self.device), p2.to(self.device), n1.to(self.device)
            loss = loss_func(p1, p2, n1)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(focal_params, 1)
            optim.step()
            pbar.set_postfix(loss=loss.item())
            outfile = get_sk_value("outfile", CONSTANTS.OUTPUT)
            if (it + 1) % checkpoint == 0 and outfile:
                torch.save(self.model.state_dict(), outfile)
