# @Filename:    custom_model.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/22/22 3:29 PM
from models.model import Model


class CustomModel(Model):
    """
    only load model
    """
    def __init__(self, path, load_method, params, dim=100,):
        self.method = load_method
        self.params = params
        super().__init__(load=True, path=path, dim=dim)

    def fit(self, iid, dataset, workers=4):
        print("DummyModel: fit")
        pass

    def save(self, path):
        print("DummyModel: save")
        pass

    def load(self, path):
        self._model = self.method(path, **self.params)

