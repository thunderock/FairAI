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
# import wandb
# wandb.login()

class FairnessAwareModel(Model):

    def __init__(self, device, num_nodes, dim, params):
        super().__init__(dim=100, load=False)
        self.model = gravlearn.Word2Vec(vocab_size=num_nodes, dim=dim)
        self.device = device
        self.params = params["params"]
        self.output = params["output"]
        self.outfile = get_sk_value("outfile", self.output)
        # self.run = wandb.init(settings=wandb.Settings(start_method='fork'), project="FairAI")

    def __del__(self):
        # self.run.finish()
        pass

    def save(self, path, biased_wv=None, kv_path=None):
        assert not bool(biased_wv) ^ bool(kv_path), "both of biased_wv and kv_path must be specified"
        torch.save(self.model.state_dict(), path)
        if biased_wv is None:
            return
        in_vec = self.model.ivectors.weight.detach().cpu().numpy()
        kv = gensim.models.KeyedVectors(in_vec.shape[1])
        kv.add_vectors(biased_wv.index_to_key, in_vec)
        kv.save(kv_path)

    def fit(self, dataset: gravlearn.DataLoader, workers=4, iid=None):
        assert not iid, 'iid not yet supported in FairnessAwareModel'
        self.device = next(self.model.parameters()).device
        self.model.train()
        self.model = self.model.to(self.device)
        self.device = next(self.model.parameters()).device
        loss_func = gravlearn.TripletLoss(embedding=self.model, dist_metric=get_sk_value(CONSTANTS.DIST_METRIC.__name__.lower(), self.params, object=True))
        checkpoint = get_sk_value("checkpoint", self.params)
        focal_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optim = torch.optim.AdamW(focal_params, lr=get_sk_value("lr", self.params))
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
            # wandb.log({"loss": loss.item()})
            if (it + 1) % checkpoint == 0 and self.outfile:
                torch.save(self.model.state_dict(), self.outfile)