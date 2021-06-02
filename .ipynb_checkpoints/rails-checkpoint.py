from aise import AISE
import torch
import torch.nn as nn
from collections import deque
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RAILS:
    def __init__(self, model, configs, x_train, y_train, batch_size=512):
        self.configs = configs
        self.aise_params = self.configs.get("aise_params", None)
        self.start_layer = self.configs.get("start_layer", -1)
        self.n_class = self.configs.get("n_class", 10)
        self._model = self.reconstruct_model(model, self.start_layer)
        self.batch_size = batch_size
        with torch.no_grad():
            self.x_train = torch.cat([
                self._model.to_start(x_train[i:i + self.batch_size].to(DEVICE)).cpu()
                for i in range(0, x_train.size(0), self.batch_size)
            ], dim=0)
        self.y_train = y_train
        self.aises = [
            AISE(model=self._model, x_orig=self.x_train, y_orig=self.y_train, dataset = "cifar", **params)
            for params in self.aise_params
        ]

    def reconstruct_model(self, model, start_layer):

        class InternalModel(nn.Module):
            def __init__(self, model, start_layer=-1):
                super(InternalModel, self).__init__()
                self._model = model
                self.start_layer = start_layer
                self.feature_mappings = deque(
                    mod[1] for mod in self._model.named_children()
                    if not ("feature" in mod[0] or "classifier" in mod[0])
                )
                self.n_layers = len(self.feature_mappings)

                self.to_start = nn.Sequential()
                if hasattr(model, "feature"):
                    self.to_start.add_module(model.feature)
                for i in range(start_layer + 1):
                    self.to_start.add_module(
                        f"pre_start_layer{i}", self.feature_mappings.popleft()
                    )

                self.hidden_layers = range(self.n_layers-self.start_layer-1)

                self.truncated_forwards = [nn.Identity()]
                self.truncated_forwards.extend([
                    self._customize_mapping(hidden_layer)
                    for hidden_layer in self.hidden_layers
                ])

            def _customize_mapping(self, end_layer=None):
                feature_mappings = list(self.feature_mappings)[:end_layer + 1]

                def truncated_forward(x):
                    for map in feature_mappings:
                        x = map(x)
                    return x

                return truncated_forward

            def truncated_forward(self, hidden_layer):
                return self.truncated_forwards[hidden_layer - self.start_layer]

        return InternalModel(model, start_layer)

    def predict(self, x):
        with torch.no_grad():
            x_start = torch.cat([
                self._model.to_start(x[i:i + self.batch_size].to(DEVICE)).cpu()
                for i in range(0, x.size(0), self.batch_size)
            ], dim=0)
        pred = np.zeros((x_start.size(0), self.n_class))
        for aise in self.aises:
            pred = pred + aise(x_start)
        return pred
