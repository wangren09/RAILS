import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from sklearn.neighbors import NearestNeighbors
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)


class GenAdapt:
    """
    core component of AISE B-cell generation
    """

    def __init__(self, mut_range, mut_prob, mode='random'):
        self.mut_range = mut_range
        self.mut_prob = mut_prob
        self.mode = mode

    def crossover(self, base1, base2, select_prob):
        assert base1.ndim == 2 and base2.ndim == 2, "Number of dimensions should be 2"
        crossover_mask = torch.rand_like(base1) < select_prob[:, None]
        return torch.where(crossover_mask, base1, base2)

    def mutate_random(self, base):
        mut = 2 * torch.rand_like(base) - 1  # uniform (-1,1)
        mut = self.mut_range * mut
        mut_mask = torch.rand_like(base) < self.mut_prob
        child = torch.where(mut_mask, base, base + mut)
        return torch.clamp(child, 0, 1)

    def crossover_complete(self, parents, select_prob):
        parent1, parent2 = parents
        child = self.crossover(parent1, parent2, select_prob)
        child = self.mutate_random(child)
        return child

    def __call__(self, *args):
        if self.mode == "random":
            base, *_ = args
            return self.mutate_random(base)
        elif self.mode == "crossover":
            assert len(args) == 2
            parents, select_prob = args
            return self.crossover_complete(parents, select_prob)
        else:
            raise ValueError("Unsupported mutation type!")


class L2NearestNeighbors(NearestNeighbors):
    """
    compatible query object class for euclidean distance
    """

    def __call__(self, X):
        return self.kneighbors(X, return_distance=False)


def neg_l2_dist(x, y):
    """
    x: (1,n_feature)
    y: (N,n_feature)
    """
    return -(x - y).pow(2).sum(dim=1).sqrt()


def inner_product(X, Y):
    return (X @ Y.T)[0]


class AISE:
    """
    implement the Adaptive Immune System Emulation
    """

    def __init__(self, x_orig, y_orig, hidden_layer=None, model=None, input_shape=None, device=torch.device("cuda"),
                 n_class=10, n_neighbors=10, query_class="l2", norm_order=2, normalize=False,
                 avg_channel=False, fitness_function="negative l2", sampling_temperature=.3, adaptive_temp=False,
                 max_generation=50, requires_init=True, apply_bound="none", c=1.0,
                 mut_range=(.05, .15), mut_prob=(.05, .15), mut_mode="crossover",
                 decay=(.9, .9), n_population=1000, memory_threshold=.25, plasma_threshold=.05,
                 keep_memory=False, return_log=False):

        self.model = model
        self.device = device

        self.x_orig = x_orig
        self.y_orig = y_orig

        if input_shape is None:
            try:
                self.input_shape = tuple(self.x_orig.shape[1:])  # mnist: (1,28,28)
            except AttributeError:
                logger.warning("Invalid data type for x_orig!")
        else:
            self.input_shape = input_shape

        self.hidden_layer = hidden_layer

        self.n_class = n_class
        self.n_neighbors = n_neighbors
        self.query_class = query_class
        self.norm_order = norm_order
        self.normalize = normalize
        self.avg_channel = avg_channel
        self.fitness_func = self._get_fitness_func(fitness_function)
        self.sampl_temp = sampling_temperature
        self.adaptive_temp = adaptive_temp

        self.max_generation = max_generation
        self.n_population = self.n_class * self.n_neighbors
        self.requires_init = requires_init
        self.apply_bound = apply_bound
        self.c = c

        self.mut_range = mut_range
        self.mut_prob = mut_prob

        if isinstance(mut_range, float):
            self.mut_range = (mut_range, mut_range)
        if isinstance(mut_prob, float):
            self.mut_prob = (mut_prob, mut_prob)

        self.mut_mode = mut_mode
        self.decay = decay
        self.n_population = n_population
        self.n_plasma = int(plasma_threshold * self.n_population)
        self.n_memory = int(memory_threshold * self.n_population) - self.n_plasma

        self.keep_memory = keep_memory
        self.return_log = return_log

        try:
            self.model.to(self.device)
            self.model.eval()
        except AttributeError:
            logger.warning("Invalid model!")

        try:
            self._query_objects = self._build_all_query_objects()
        except:
            logger.warning("Cannot build query objects!")

    @staticmethod
    def _get_fitness_func(func_str):
        if func_str == "negative l2":
            return neg_l2_dist
        elif func_str == "inner product":
            return inner_product

    def _build_class_query_object(self, xh_orig, class_label=-1):
        if class_label + 1:
            x_class = xh_orig[self.y_orig == class_label]
        else:
            x_class = xh_orig
        if self.query_class == "l2":
            query_object = L2NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1).fit(x_class)
        return query_object

    def _build_all_query_objects(self):
        xh_orig = self._hidden_repr_mapping(self.x_orig, query=True).detach().cpu().numpy()
        if self.adaptive_temp:
            self.sampl_temp *= np.sqrt(xh_orig.shape[1] / np.prod(
                self.input_shape)).item()  # heuristic sampling temperature: proportion to the square root of feature space dimension
        if self.n_class:
            logger.info("Building query objects for {} classes {} samples...".format(self.n_class, self.x_orig.size(0)))
            query_objects = [self._build_class_query_object(xh_orig,class_label=i) for i in range(self.n_class)]
        else:
            logger.info("Building one single query object {} samples...".format(self.x_orig.size(0)))
            query_objects = [self._build_class_query_object(xh_orig)]
        return query_objects

    def _query_nns_ind(self, Q):
        assert Q.ndim == 2, "Q: 2d array-like (n_queries,n_features)"
        if self.n_class:
            logger.info("Searching {} naive B cells per class for each of {} antigens...".format(self.n_neighbors, len(Q)))
            rel_ind = [query_obj(Q) for query_obj in self._query_objects]
            abs_ind = []
            for c in range(self.n_class):
                class_ind = np.where(self.y_orig.numpy() == c)[0]
                abs_ind.append(class_ind[rel_ind[c]])
        else:
            logger.info("Searching {} naive B cells for each of {} antigens...".format(self.n_neighbors, Q.size(0)))
            abs_ind = [query_obj(Q) for query_obj in self._query_objects]
        return np.concatenate(abs_ind,axis=1)

    def _hidden_repr_mapping(self, x, batch_size=2048, query=False):
        """
        transform b cells and antigens into inner representations of AISE
        """
        if self.hidden_layer is not None:
            xhs = []
            for i in range(0, x.size(0), batch_size):
                xx = x[i:i + batch_size]
                with torch.no_grad():
                    if query:
                        xh = self.model.truncated_forward(self.hidden_layer)(xx.to(self.device)).detach().cpu()
                    else:
                        xh = self.model.truncated_forward(self.hidden_layer)(xx.to(self.device))
                    if self.avg_channel:
                        xh = xh.sum(dim=1)
                    xh = xh.flatten(start_dim=1)
                    if self.normalize:
                        xh = xh / xh.pow(2).sum(dim=1, keepdim=True).sqrt()
                    xhs.append(xh.detach())
            return torch.cat(xhs)
        else:
            xh = x.flatten(start_dim=1)
            if self.normalize:
                xh = xh / xh.pow(2).sum(dim=1, keepdim=True).sqrt()
            return xh.detach()

    @staticmethod
    def clip_class_bound(self, x, y, class_center, class_bound):
        return torch.min(torch.max(x, (class_center - class_bound)[y]), (class_center + class_bound)[y])

    def generate_b_cells(self, ant, ant_tran, nbc_ind, y_ant=None):
        assert ant_tran.ndim == 2, "ant: 2d tensor (n_antigens,n_features)"
        pla_bcs = []
        pla_labs = []
        if self.keep_memory:
            mem_bcs = []
            mem_labs = []
        else:
            mem_bcs = None
            mem_labs = None
        logger.info("Affinity maturation process starts with population of {}...".format(self.n_population))
        if self.return_log:
            ant_logs = []  # store the history dict in terms of metrics for antigens
        else:
            ant_logs = None

        for n in range(ant.size(0)):
            genadapt = GenAdapt(self.mut_range[1], self.mut_prob[1], mode=self.mut_mode)
            curr_gen = torch.Tensor(self.x_orig[nbc_ind[n]]).flatten(start_dim=1).to(self.device)  # naive b cells
            labels = torch.LongTensor(self.y_orig[nbc_ind[n]]).to(self.device)
            if self.apply_bound != "none":
                class_center = []
                if self.apply_bound == "hard":
                    class_bound = []
                for i in range(0, len(curr_gen), self.n_neighbors):
                    class_center.append(torch.mean(curr_gen[i:i + self.n_neighbors], dim=0))
                    if self.apply_bound == "hard":
                        class_bound.append((curr_gen[i:i + self.n_neighbors] - class_center[-1]).abs().max(dim=0)[0])
                class_center = torch.stack(class_center)
                if self.apply_bound == "hard":
                    class_bound = torch.stack(class_bound)
            if self.requires_init:
                assert self.n_population % (
                        self.n_class * self.n_neighbors) == 0, \
                    "n_population should be divisible by the product of n_class and n_neighbors"
                curr_gen = curr_gen.repeat((self.n_population // (self.n_class * self.n_neighbors), 1))
                curr_gen = genadapt.mutate_random(curr_gen)  # initialize *NOTE: torch.Tensor.repeat <> numpy.repeat
                labels = labels.repeat(self.n_population // (self.n_class * self.n_neighbors))
                if self.apply_bound == "hard":
                    curr_gen = self.clip_class_bound(curr_gen, labels, class_center, class_bound)
            curr_repr = self._hidden_repr_mapping(curr_gen.view((-1,) + self.x_orig.size()[1:]))
            fitness_score = self.fitness_func(ant_tran[n].unsqueeze(0).to(self.device), curr_repr.to(self.device))
            if self.apply_bound == "soft":
                fitness_score = fitness_score + self.c * F.pairwise_distance(curr_gen, class_center[labels])
            best_pop_fitness = float('-inf')
            decay_coef = (1., 1.)
            num_plateau = 0
            ant_log = dict()  # history log for each antigen
            # zeroth generation logging
            if self.return_log:
                fitness_pop_hist = []
                pop_fitness = fitness_score.sum().item()
                fitness_pop_hist.append(pop_fitness)
                if y_ant is not None:
                    fitness_true_class_hist = []
                    pct_true_class_hist = []
                    true_class_fitness = fitness_score[labels == y_ant[n]].sum().item()
                    fitness_true_class_hist.append(true_class_fitness)
                    true_class_pct = (labels == y_ant[n]).astype('float').mean().item()
                    pct_true_class_hist.append(true_class_pct)

            # active classes in the initial generation
            class_vector = torch.arange(self.n_class)[:,None].to(self.device)

            for i in range(self.max_generation):
                survival_prob = F.softmax(fitness_score / self.sampl_temp, dim=-1)
                parents_ind1 = Categorical(probs=survival_prob).sample((self.n_population,))

                parent_pos = (labels[parents_ind1]==class_vector)
                class_count = parent_pos.sum(dim=1)
                active_class = (class_count>0) # active classes in the current generation
                class_vector = class_vector[active_class]

                # check homogeneity
                if len(class_vector) == 1:
                    break # early stop

                if self.mut_mode == "crossover":
                    parents_ind2 = torch.zeros_like(parents_ind1) # second parents

                    parent_pos = parent_pos[active_class]
                    class_pos = (labels==class_vector)
                    class_probs = survival_prob.where(class_pos,torch.zeros_like(survival_prob))

                    for pos,probs,cnt in zip(parent_pos,class_probs,class_count[active_class]):
                        parents_ind2_class = Categorical(probs=probs).sample((cnt,))
                        parents_ind2[pos] = parents_ind2_class
                    parent_pairs = [curr_gen[parents_ind1],curr_gen[parents_ind2]]
                    # crossover between two parents
                    curr_gen = genadapt(parent_pairs, fitness_score[parents_ind1] /\
                                      (fitness_score[parents_ind1]+fitness_score[parents_ind2]))
                else:
                    parents = curr_gen[parents_ind1]
                    curr_gen = genadapt(parents)

                if self.apply_bound == "hard":
                    curr_gen = self.clip_class_bound(curr_gen, labels, class_center, class_bound)

                curr_repr = self._hidden_repr_mapping(curr_gen.view((-1,) + self.x_orig.size()[1:]))
                labels = labels[parents_ind1.cpu()]

                fitness_score = self.fitness_func(ant_tran[n].unsqueeze(0).to(self.device), curr_repr.to(self.device))
                if self.apply_bound == "soft":
                    fitness_score = fitness_score + self.c * F.pairwise_distance(curr_gen, class_center[labels])
                pop_fitness = fitness_score.sum().item()

                if self.return_log:
                    # logging
                    fitness_pop_hist.append(pop_fitness)
                    if y_ant is not None:
                        true_class_fitness = fitness_score[labels == y_ant[n]].sum().item()
                        fitness_true_class_hist.append(true_class_fitness)
                        true_class_pct = (labels == y_ant[n]).astype('float').mean().item()
                        pct_true_class_hist.append(true_class_pct)

                # adaptive shrinkage of certain hyper-parameters
                if self.decay:
                    assert len(self.decay) == 2
                    if pop_fitness < best_pop_fitness:
                        if num_plateau >= max(math.log(self.mut_range[0] / self.mut_range[1], self.decay[0]),
                                              math.log(self.mut_prob[0] / self.mut_prob[1], self.decay[1])):
                            # early stop
                            break
                        decay_coef = tuple(decay_coef[i] * self.decay[i] for i in range(2))
                        num_plateau += 1
                        genadapt = GenAdapt(max(self.mut_range[0], self.mut_range[1] * decay_coef[0]),
                                            max(self.mut_prob[0], self.mut_prob[1] * decay_coef[1]),
                                            mode=self.mut_mode)
                    else:
                        best_pop_fitness = pop_fitness
            _, fitness_rank = torch.sort(fitness_score.cpu())
            if self.return_log:
                ant_log["fitness_pop"] = fitness_pop_hist
                if y_ant is not None:
                    ant_log["fitness_true_class"] = fitness_true_class_hist
                    ant_log["pct_true_class"] = pct_true_class_hist
            pla_bcs.append(curr_gen[fitness_rank[-self.n_plasma:]].detach().cpu())
            pla_labs.append(labels[fitness_rank[-self.n_plasma:]].cpu().numpy())
            if self.keep_memory:
                mem_bcs.append(curr_gen[fitness_rank[-(self.n_memory+self.n_plasma):-self.n_plasma]].detach().cpu())
                mem_labs.append(labels[fitness_rank[-(self.n_memory+self.n_plasma):-self.n_plasma]].cpu().numpy())
            if self.return_log:
                ant_logs.append(ant_log)

        pla_bcs = torch.stack(pla_bcs).view((-1,self.n_plasma)+self.input_shape).numpy()
        pla_labs = np.stack(pla_labs).astype(np.int)
        if self.keep_memory:
            mem_bcs = torch.stack(mem_bcs).view((-1, self.n_mem) + self.input_shape).numpy()
            mem_labs = np.stack(mem_labs).astype(np.int)

        return mem_bcs, mem_labs, pla_bcs, pla_labs, ant_logs

    def clonal_expansion(self, ant, y_ant=None):
        logger.info("Clonal expansion starts...")
        ant_tran = self._hidden_repr_mapping(ant.detach())
        nbc_ind = self._query_nns_ind(ant_tran.detach().cpu().numpy())
        mem_bcs, mem_labs, pla_bcs, pla_labs, ant_logs = self.generate_b_cells(ant.flatten(start_dim=1), ant_tran, nbc_ind)
        if self.keep_memory:
            logger.info("{} plasma B cells and {} memory generated!".format(pla_bcs.shape[0]*self.n_plasma, mem_bcs.shape[0]*self.n_memory))
        else:
            logger.info("{} plasma B cells generated!".format(pla_bcs.shape[0]*self.n_plasma))
        return mem_bcs, mem_labs, pla_bcs, pla_labs, ant_logs

    def __call__(self, ant):
        _, _, _, pla_labs, *_ = self.clonal_expansion(ant)
        # output the prediction of aise
        return AISE.predict_proba(pla_labs, self.n_class)

    @staticmethod
    def predict(labs, n_class):
        return AISE.predict_proba(labs, n_class).argmax(axis=1)

    @staticmethod
    def predict_proba(labs, n_class):
        return np.stack(list(map(lambda x: np.bincount(x, minlength=n_class) / x.size, labs)))
