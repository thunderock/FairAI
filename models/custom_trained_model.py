# @Filename:    custom_trained_model.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/22/22 3:29 PM
from models.model import Model
from utils.model_utils import identity


class CustomTrainedModel(Model):
    """
    only load model
    """
    def __init__(self, path, load_method, load_params, dim=100, ):
        self.load_method = load_method
        self.load_params = load_params
        super().__init__(load=True, path=path, dim=dim)

    def fit(self, iid, dataset, workers=4):
        print("DummyModel: fit")
        pass

    def save(self, path):
        print("DummyModel: save")
        pass

    def load(self, path):
        self._model = self.load_method(path, **self.load_params)

