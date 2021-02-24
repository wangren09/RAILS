import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from functools import reduce


class DKNNAttack:

    def __init__(
            self,
            model,
            train_data=None,
            train_targets=None,
            dknn=None,
            eta=None,
            alpha=0.1,
            eps=8 / 255.,
            lr=0.1,
            max_iter=20,
            random_init=True,
            batch_size=16,
            hidden_layers=-1,
            n_class=10,
            k_neighbors=5,
            m_neighborhood=15,
            device=torch.device("cpu")
    ):
        self.lr = lr
        self.eps = eps
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_init = random_init
        self.batch_size = batch_size

        self.hidden_layers = hidden_layers
        self._model = self._wrap_model(model)
        self.hidden_layers = self._model.hidden_layers
        self.device = device
        self._model.eval()  # make sure the model is run in the eval mode
        self._model.to(self.device)

        if dknn is None:
            self.train_data = train_data
            self.train_targets = np.array(train_targets)
            self.hidden_layers = self._model.hidden_layers
            self.n_class = n_class
            self.k_neighbors = k_neighbors
        else:
            assert dknn.hidden_layers == self.hidden_layers,\
                "Hidden layers of dknn and attack must agree!"
            self.dknn = dknn
            for name, value in dknn.__dict__.items():
                if name in ("_model", "hidden_layers"):
                    continue
                if name == "n_neighbors":
                    setattr(self, "k_neighbors", value)
                if name == "_nns":
                    setattr(self, "_knn_clfs", value)
                else:
                    setattr(self, name, value)

        self.m_neighborhood = m_neighborhood
        self.train_targets_class = [train_targets == i for i in range(self.n_class)]

        if eta is None:
            self._nn_clfs, self.eta = self._build_nn_clfs(return_eta=True)
        else:
            self._nn_clfs, _ = self._build_nn_clfs()

    def _get_hidden_repr(self, x, return_targets=False, batch_size=256):

        hidden_reprs = []
        targets = None
        if return_targets:
            outs = []

        for i in range(0, x.size(0), batch_size):
            x_batch = x[i:i + batch_size]
            if return_targets:
                hidden_reprs_batch, outs_batch = self._model(x_batch.to(self.device))
            else:
                hidden_reprs_batch, _ = self._model(x_batch.to(self.device))
            hidden_reprs_batch = list(map(
                lambda x: x.detach().cpu().flatten(start_dim=1),
                hidden_reprs_batch
            ))
            hidden_reprs.append(hidden_reprs_batch)
            if return_targets:
                outs.append(outs_batch)

        hidden_reprs = [
            np.concatenate([hidden_batch[i] for hidden_batch in hidden_reprs], axis=0)
            for i in range(len(self.hidden_layers))
        ]

        if return_targets:
            outs = np.concatenate(outs, axis=0)
            targets = outs.argmax(axis=1)

        return hidden_reprs, targets

    def _build_nn_clfs(self, sample_size=0.2, random_state=42, return_eta=False):

        with torch.no_grad():
            hidden_reprs, _ = self._get_hidden_repr(self.train_data)
        nn_clfs = [
            [
                NearestNeighbors(n_neighbors=self.m_neighborhood, n_jobs=-1).fit(hidden_repr[class_inds])
                for class_inds in self.train_targets_class
            ]
            for hidden_repr in tqdm(hidden_reprs)
        ]
        if not hasattr(self, "_knn_clfs"):
            self._knn_clfs = [
                NearestNeighbors(n_neighbors=self.k_neighbors, n_jobs=-1).fit(hidden_repr)
                for hidden_repr in hidden_reprs
            ]
        if return_eta:
            train_size = self.train_data.shape[0]
            np.random.seed(random_state)
            train_sample_indices = torch.LongTensor(np.random.choice(
                np.arange(train_size),
                size=int(sample_size * train_size),
                replace=False
            ))
            hidden_reprs = [hidden_repr[train_sample_indices] for hidden_repr in hidden_reprs]
            eta = [
                knn_clf.kneighbors(hidden_repr, return_distance=True)[0].max(axis=-1).mean()
                for knn_clf, hidden_repr in zip(self._knn_clfs, hidden_reprs)
            ]
            return nn_clfs, eta
        else:
            return nn_clfs

    def _wrap_model(self, model):

        class ModelWrapper(nn.Module):

            def __init__(self, model, hidden_layers):
                super(ModelWrapper, self).__init__()
                self._model = model
                self.hidden_mappings = [
                    m[1] for m in model.named_children()
                    if isinstance(m[1], nn.Sequential) and "classifier" not in m[0]
                ]
                if hidden_layers == -1:
                    self.hidden_layers = list(range(len(self.hidden_mappings)))
                else:
                    self.hidden_layers = hidden_layers
                self.classifier = self._model.classifier

            def forward(self, x):
                hidden_reprs = []
                for mp in self.hidden_mappings:
                    x = mp(x)
                    hidden_reprs.append(x)
                out = self.classifier(x.flatten(start_dim=1))
                return [hidden_reprs[i] for i in self.hidden_layers], out

            def forward_branch(self, hidden_layer):

                hidden_mappings = self.hidden_mappings[:hidden_layer + 1]

                def branch(x):
                    for mp in hidden_mappings:
                        x = mp(x)
                    return x.detach()

                return branch

        return ModelWrapper(model, self.hidden_layers)

    def knn_loss(self, l, hidden_repr, nn_repr, n_samples):
        eta = torch.LongTensor(self.eta).to(self.device)
        l2dist = (
                hidden_repr.reshape(n_samples, 1, -1) - nn_repr.reshape(n_samples, self.m_neighborhood, -1)
        ).pow(2).sum(dim=-1).sqrt()
        return (torch.sigmoid(self.alpha * (l2dist - eta[l]))).sum()

    def attack(
            self,
            x,
            params,
            bounds,
            optimizer,
            nn_reprs=None
    ):

        pert = (bounds[1] - bounds[0]) * (1 + torch.tanh(params)) / 2 + bounds[0] - x
        hidden_reprs, _ = self._model(x + pert)

        loss = 0
        for l, (hidden_repr, nn_repr) in enumerate(zip(hidden_reprs, nn_reprs)):
            loss += self.knn_loss(l, hidden_repr, nn_repr, x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return params

    def _get_targets(
            self,
            y,
            nn_distances,
            knn_indices
    ):

        knn_labels = np.concatenate([self.train_targets[knn_inds] for knn_inds in knn_indices], axis=1)
        knn_pred = np.stack(list(map(
            lambda x: np.bincount(x, minlength=self.n_class),
            knn_labels
        ))).argmax(axis=1)
        to_attack = knn_pred == y.cpu().numpy()
        dist_rank = np.argsort(reduce(lambda x, y: x + y, nn_distances), axis=1)
        targets = np.where(dist_rank[:, 0] != y.cpu().numpy(), dist_rank[:, 0], dist_rank[:, 1])

        return (
            torch.LongTensor(targets).to(self.device),
            torch.BoolTensor(to_attack).to(self.device)
        )

    def generate(
            self,
            x,
            y=None
    ):

        x_adv = []
        for i in range(0, x.size(0), self.batch_size):
            x_batch = x[i: i + self.batch_size].to(self.device)
            if y is None:
                y_batch = self._model(x_batch)
                if isinstance(y_batch, (tuple, list)):
                    y_batch = y_batch[-1]
                y_batch = y_batch.max(dim=-1)[1].to(self.device)
            else:
                y_batch = y[i: i + self.batch_size].to(self.device)

            hidden_reprs_batch, _ = self._model(x_batch)

            nn_distances_batch = []
            nn_indices_batch = []
            knn_indices_batch = []
            for knn_clf, nn_clf, hidden_repr in zip(self._knn_clfs, self._nn_clfs, hidden_reprs_batch):
                dists = []
                inds = []
                for i, clf in enumerate(nn_clf):
                    dist, ind = clf.kneighbors(
                        hidden_repr.flatten(start_dim=1).detach().cpu().numpy(),
                        return_distance=True
                    )
                    dists.append(dist.mean(axis=1))
                    inds.append(np.where(self.train_targets_class[i])[0][ind])
                nn_distances_batch.append(np.stack(dists, axis=1))
                nn_indices_batch.append(torch.LongTensor(np.stack(inds, axis=1)).to(self.device))
                knn_indices_batch.append(knn_clf.kneighbors(
                    hidden_repr.flatten(start_dim=1).detach().cpu().numpy(),
                    return_distance=False
                ))
            targets_batch, to_attack_batch = self._get_targets(
                y_batch,
                nn_distances=nn_distances_batch,
                knn_indices=knn_indices_batch
            )
            nn_indices_batch = [
                nn_inds[range(len(nn_indices_batch)), targets_batch, :]
                for nn_inds in nn_indices_batch
            ]
            if to_attack_batch.numel():
                with torch.no_grad():
                    nn_reprs = [
                        self._model.forward_branch(l)(
                            self.train_data[nn_inds[to_attack_batch].cpu().flatten()].to(self.device)
                        )
                        for l, nn_inds in zip(self.hidden_layers, nn_indices_batch)
                    ]
                if self.random_init:
                    init_params = 0.1 * torch.randn_like(x_batch[to_attack_batch])
                else:
                    init_params = torch.zeros_like(x_batch[to_attack_batch])

                bounds = (
                    torch.maximum(torch.tensor(0).to(self.device), x_batch[to_attack_batch] - self.eps),
                    torch.minimum(torch.tensor(1).to(self.device), x_batch[to_attack_batch] + self.eps)
                )

                params = nn.Parameter(
                    init_params,
                    requires_grad=True
                )
                optimizer = Adam([params], lr=self.lr)

                for j in range(self.max_iter):
                    self.attack(
                        x_batch[to_attack_batch],
                        params,
                        bounds,
                        optimizer,
                        nn_reprs=nn_reprs
                    )
                x_adv_batch = x_batch
                x_adv_batch[to_attack_batch] = (bounds[1] - bounds[0]) * (1 + torch.tanh(params.data)) / 2 + bounds[0]
                x_adv.append(x_adv_batch.cpu())
            else:
                x_adv.append(x_batch.cpu())
        return torch.cat(x_adv, dim=0)


if __name__ == "__main__":
    import os
    from dknn import DKNN
    from models.vgg import VGG16
    from torchvision.datasets import CIFAR10

    ROOT = "../datasets"
    MODEL_WEIGHTS = "../model_weights/cifar_vggrob.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DOWNLOAD = not os.path.exists("../datasets/cifar-10-python.tar.gz")

    trainset = CIFAR10(root=ROOT, train=True, download=DOWNLOAD)
    testset = CIFAR10(root=ROOT, train=False, download=DOWNLOAD)
    train_data, train_targets = (
        torch.FloatTensor(trainset.data.transpose(0, 3, 1, 2) / 255.)[:2000],
        torch.LongTensor(trainset.targets)[:2000]
    )  # for memory's sake, only take 2000 as train set

    model = VGG16()
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    dknn = DKNN(
        model,
        train_data,
        train_targets,
        hidden_layers=[4],
        device=DEVICE
    )

    x, y = (
        torch.FloatTensor(testset.data.transpose(0, 3, 1, 2) / 255.)[:256],
        torch.LongTensor(testset.targets)[:256]
    )  # for memory's sake, only take 256 as test set

    x_adv = DKNNAttack(
        model,
        train_data,
        train_targets,
        dknn,
        hidden_layers=[4],
        device=DEVICE
    ).generate(x, y)

    pred_benign = dknn(x.to(DEVICE)).argmax(axis=1)
    acc_benign = (pred_benign == y.numpy()).astype("float").mean()
    print(f"The benign accuracy is {acc_benign}")

    pred_adv = dknn(x_adv.to(DEVICE)).argmax(axis=1)
    acc_adv = (pred_adv == y.numpy()).astype("float").mean()
    print(f"The adversarial accuracy is {acc_adv}")
