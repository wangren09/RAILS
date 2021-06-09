import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
#from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex
#import random
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)


class GeneticOperator:
    """
    Genetic Operator
    """

    def __init__(self, mut_range, mut_prob, cliprg, type='mutate'):
        self.mut_range = mut_range
        self.mut_prob = mut_prob
        self.cliprg = cliprg
        self.type = type

    def crossover(self, base1, base2, select_prob):
        crossover_mask = torch.rand_like(base1) < select_prob[:,None,None,None]
        return torch.where(crossover_mask, base1, base2)

    def mutate(self, base, clrg):
        mut = 2 * torch.rand_like(base) - 1  # uniform (-1,1)
        mut = self.mut_range * mut
        mut_mask = torch.rand_like(base) < self.mut_prob
        child = torch.where(mut_mask, base, base + mut)
        return torch.clamp(child, 0, clrg)

    def crossover_with_mutation(self, parents, select_prob):
        parent1, parent2 = parents
        child = self.crossover(parent1, parent2, select_prob)
        child = self.mutate(child, self.cliprg)
        return child

    def __call__(self, *args):
        if self.type == "mutate":
            base, *_ = args
            return self.mutate(base, self.cliprg)
        elif self.type == "crossover":
            assert len(args) == 2
            parents, select_prob = args
            return self.crossover_with_mutation(parents, select_prob)
        else:
            raise ValueError("Unsupported operator type!")


# class L2NearestNeighbors(NearestNeighbors):
#     """
#     compatible query object class for euclidean distance
#     """

#     def __call__(self, X):
#         return self.kneighbors(X, return_distance=False)


def neg_l2_dist(x, y):
    return -(x - y).pow(2).sum(dim=-1).sqrt()


def inner_product(X, Y):
    return (X @ Y.T)[0]


class AISE:
    """
    Adaptive Immune System Emulator for RAILS
    """

    def __init__(
            self,
            x_orig,
            y_orig,
            dataset = "cifar",
            hidden_layer=None,
            model=None,
            input_shape=None,
            device=torch.device("cuda"),
            n_class=10,
            n_neighbors=10,
            query_class="l2",
            norm_order=2,
            normalize=False,
            avg_channel=False,
            fitness_function="negative l2",
            sampling_temperature=.3,
            adaptive_temp=False,
            max_generation=10,
            requires_init=True,
            mut_range=(.05, .15),
            mut_prob=(.005, .015),
            genop_type="crossover",
            decay=(.9, .9),
            n_population=1000,
            memory_threshold=.25,
            plasma_threshold=.05,
            keep_memory=False,
            return_log=False,
            return_bcells=False
    ):

        self.model = model
        self.device = device

        self.x_orig = x_orig
        self.y_orig = y_orig

        if input_shape is None:
            try:
                self.input_shape = self.x_orig.shape[1:]  # mnist: (1,28,28)
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
        if dataset == "cifar":
            self.cliprg = 5
        else:
            self.cliprg = 1

        self.max_generation = max_generation
        self.requires_init = requires_init

        self.mut_range = mut_range
        self.mut_prob = mut_prob
        self.genop_type = genop_type

        if isinstance(mut_range, float):
            self.mut_range = (mut_range, mut_range)
        if isinstance(mut_prob, float):
            self.mut_prob = (mut_prob, mut_prob)

        self.decay = decay
        self.n_population = n_population
        self.n_plasma = int(plasma_threshold * self.n_population)
        self.n_memory = int(memory_threshold * self.n_population) - self.n_plasma

        self.keep_memory = keep_memory
        self.return_log = return_log
    
        self.return_bcells = return_bcells
    
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
            #query_object = L2NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1).fit(x_class)
            f = len(x_class[0])
            query_object = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed
            for i in range(len(x_class)):
                query_object.add_item(i, x_class[i])
            query_object.build(n_trees=100, n_jobs=-1)
        return query_object

    def _build_all_query_objects(self):
        xh_orig = self._hidden_repr_mapping(self.x_orig, query=True).detach().cpu().numpy()
        # heuristic sampling temperature: proportion to the square root of feature space dimension
        if self.adaptive_temp:
            self.sampl_temp *= np.sqrt(xh_orig.shape[1] / np.prod(self.input_shape)).item()
        if self.n_class:
            logger.info("Building query objects for {} classes {} samples...".format(self.n_class, self.x_orig.size(0)))
            query_objects = [self._build_class_query_object(xh_orig, class_label=i) for i in range(self.n_class)]
        else:
            logger.info("Building one single query object {} samples...".format(self.x_orig.size(0)))
            query_objects = [self._build_class_query_object(xh_orig)]
        return query_objects

    def _query_nns_ind(self, Q):
        assert Q.ndim == 2, "Q: 2d array-like (n_queries,n_features)"
        if self.n_class:
            logger.info(
                "Searching {} naive B cells per class for each of {} antigens...".format(self.n_neighbors, len(Q)))
            #rel_ind = [query_obj(Q) for query_obj in self._query_objects]
            rel_ind = [[query_obj.get_nns_by_vector(Q[j], self.n_neighbors, search_k=-1, include_distances=False) for j in range(len(Q))] for query_obj in self._query_objects]
            abs_ind = []
            for c in range(self.n_class):
                class_ind = np.where(self.y_orig.numpy() == c)[0]
                abs_ind.append(class_ind[rel_ind[c]])
        else:
            logger.info("Searching {} naive B cells for each of {} antigens...".format(self.n_neighbors, Q.size(0)))
            #abs_ind = [query_obj(Q) for query_obj in self._query_objects]
            abs_ind = [query_obj.get_nns_by_vector(Q, self.n_neighbors, search_k=-1, include_distances=False) for query_obj in self._query_objects]
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

    def generate_b_cells(self, ant, ant_tran, nbc_ind, y_ant=None):
        assert ant_tran.ndim == 2, "ant: 2d tensor (n_antigens,n_features)"

        # make sure data and indices are in the same class
        if isinstance(self.x_orig, torch.Tensor):
            nbc_ind = torch.LongTensor(nbc_ind)

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

        head_shape = (self.n_class, self.n_population // self.n_class)
        # static index
        static_index = torch.arange(self.n_population).reshape(head_shape).to(self.device)
        
        for n in range(ant.size(0)):
            genop = GeneticOperator(self.mut_range[1], self.mut_prob[1], self.cliprg, type=self.genop_type)
            curr_gen = torch.Tensor(self.x_orig[nbc_ind[n]]).to(self.device)  # naive b cells
            labels = torch.LongTensor(self.y_orig[nbc_ind[n]]).to(self.device)
            if self.requires_init:
                assert self.n_population % (self.n_class * self.n_neighbors) == 0, \
                    "n_population should be divisible by the product of n_class and n_neighbors"
                curr_gen = curr_gen.repeat_interleave(self.n_population // (self.n_class * self.n_neighbors), dim=0)
                curr_gen = genop.mutate(curr_gen, self.cliprg)  # initialize *NOTE: torch.Tensor.repeat <> numpy.repeat
                labels = labels.repeat_interleave(self.n_population // (self.n_class * self.n_neighbors))
            curr_repr = self._hidden_repr_mapping(curr_gen).reshape(head_shape+(-1,))
            fitness_score = self.fitness_func(ant_tran[n].to(self.device), curr_repr.to(self.device))
            _, fitness_rank = torch.sort(fitness_score.flatten().cpu())
            
            best_pop_fitness = float('-inf')
            decay_coef = (1., 1.)
            num_plateau = 0
            ant_log = dict()  # history log for each antigen
            # zeroth generation logging
            if self.return_log:
                fitness_pop_hist = []
                proba_true_class_hist = []
                sum_fitness_true_class_hist = []
                pop_fitness = torch.exp(fitness_score).sum().item()
                fitness_pop_hist.append(pop_fitness)
                if y_ant is not None:
                    fitness_true_class_hist = []
#                     true_class_fitness = fitness_score[y_ant[n]].sum().item()
                    true_class_fitness = (torch.exp(fitness_score.flatten())*(labels==y_ant[n]))[fitness_rank[-self.n_plasma:]].sum().item()
                    true_class_total = (labels[fitness_rank[-self.n_plasma:]]==y_ant[n]).sum().item()
                    if true_class_total == 0:
                        true_class_fitness = float("-Inf")
                    else:
                        true_class_fitness /= true_class_total
                    sum_fitness_true_class_hist.append(torch.exp(fitness_score[y_ant[n]]).sum().item())
                    fitness_true_class_hist.append(true_class_fitness)
                    proba_true_class_hist.append(true_class_total/self.n_plasma)

            for i in range(self.max_generation):
                survival_prob = F.softmax(fitness_score / self.sampl_temp, dim=-1)
                if self.genop_type == "crossover":
                    parent_inds = Categorical(probs=survival_prob).sample((head_shape[1], 2))
                    parent_inds1, parent_inds2 = parent_inds[:, 0, :].t(), parent_inds[:, 1, :].t()
                    parent_inds1_flat, parent_inds2_flat = static_index.gather(-1, parent_inds1).flatten(),\
                                                           static_index.gather(-1, parent_inds2).flatten()
                    parent_pairs = [curr_gen[parent_inds1_flat], curr_gen[parent_inds2_flat]]
                    # crossover between two parents
                    fitness_score_flat = fitness_score.flatten()
                    select_prob = fitness_score_flat[parent_inds1_flat] /\
                                  (fitness_score_flat[parent_inds1_flat] + fitness_score_flat[parent_inds2_flat])
                    curr_gen = genop(parent_pairs, select_prob)
                else:
                    parent_inds1 = Categorical(probs=survival_prob).sample((head_shape[1], 1))
                    parent_inds1 = parent_inds1[:, 0, :].t()
                    parent_inds_flat = static_index.gather(-1, parent_inds1).flatten()
                    curr_gen = genop(curr_gen[parent_inds_flat])

                curr_repr = self._hidden_repr_mapping(curr_gen).reshape(head_shape + (-1,))

                fitness_score = self.fitness_func(ant_tran[n].to(self.device), curr_repr.to(self.device))
                _, fitness_rank = torch.sort(fitness_score.flatten().cpu())
                pop_fitness = fitness_score.sum().item()

                if self.return_log:
                    # logging
                    fitness_pop_hist.append(pop_fitness)
                    if y_ant is not None:
#                         true_class_fitness = fitness_score[y_ant[n]].sum().item()
                        true_class_fitness = (torch.exp(fitness_score.flatten())*(labels==y_ant[n]))[fitness_rank[-self.n_plasma:]].sum().item()
                        true_class_total = (labels[fitness_rank[-self.n_plasma:]]==y_ant[n]).sum().item()
                        if true_class_total == 0:
                            true_class_fitness = float("-Inf")
                        else:
                            true_class_fitness /= true_class_total
                        sum_fitness_true_class_hist.append(torch.exp(fitness_score[y_ant[n]]).sum().item())
                        fitness_true_class_hist.append(true_class_fitness)
                        proba_true_class_hist.append(true_class_total/self.n_plasma)

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
                        genop = GeneticOperator(
                            max(self.mut_range[0], self.mut_range[1] * decay_coef[0]),
                            max(self.mut_prob[0], self.mut_prob[1] * decay_coef[1]),
                            self.cliprg,
                            type=self.genop_type
                        )
                    else:
                        best_pop_fitness = pop_fitness

            _, fitness_rank = torch.sort(fitness_score.flatten().cpu())
            if self.return_log:
                ant_log["fitness_pop"] = fitness_pop_hist
                if y_ant is not None:
                    ant_log["sum_fitness_true_class"] = sum_fitness_true_class_hist
                    ant_log["fitness_true_class"] = fitness_true_class_hist
                    ant_log["proba_true_class"] = proba_true_class_hist
            if self.return_bcells:
                pla_bcs.append(curr_gen[fitness_rank[-self.n_plasma:]].detach().cpu())
            pla_labs.append(labels[fitness_rank[-self.n_plasma:]].cpu().numpy())
            if self.keep_memory:
                if self.return_bcells:
                    mem_bcs.append(curr_gen[fitness_rank[-(self.n_memory + self.n_plasma):-self.n_plasma]].detach().cpu())
                mem_labs.append(labels[fitness_rank[-(self.n_memory + self.n_plasma):-self.n_plasma]].cpu().numpy())
            if self.return_log:
                ant_logs.append(ant_log)
        if self.return_bcells:
            pla_bcs = torch.stack(pla_bcs).view((-1, self.n_plasma) + self.input_shape).numpy()
        pla_labs = np.stack(pla_labs).astype(np.int)
        if self.keep_memory:
            if self.return_bcells:
                mem_bcs = torch.stack(mem_bcs).view((-1, self.n_mem) + self.input_shape).numpy()
            mem_labs = np.stack(mem_labs).astype(np.int)

        return mem_bcs, mem_labs, pla_bcs, pla_labs, ant_logs

    def clonal_expansion(self, ant, y_ant=None):
        logger.info("Clonal expansion starts...")
        ant_tran = self._hidden_repr_mapping(ant.detach())
        nbc_ind = self._query_nns_ind(ant_tran.detach().cpu().numpy())
        mem_bcs, mem_labs, pla_bcs, pla_labs, ant_logs = self.generate_b_cells(
            ant.flatten(start_dim=1),
            ant_tran,
            nbc_ind,
            y_ant
        )
        if self.keep_memory:
            logger.info("{} plasma B cells and {} memory generated!".format(pla_labs.shape[0] * self.n_plasma,
                                                                            mem_labs.shape[0] * self.n_memory))
        else:
            logger.info("{} plasma B cells generated!".format(pla_labs.shape[0] * self.n_plasma))
        return mem_bcs, mem_labs, pla_bcs, pla_labs, ant_logs

    def __call__(self, ant):
        _, _, _, pla_labs, *_ = self.clonal_expansion(ant)
        return AISE.predict_proba(pla_labs, self.n_class)

    @staticmethod
    def predict(labs, n_class):
        return AISE.predict_proba(labs, n_class).argmax(axis=1)

    @staticmethod
    def predict_proba(labs, n_class):
        return np.stack(list(map(lambda x: np.bincount(x, minlength=n_class) / x.size, labs)))