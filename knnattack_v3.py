import torch
from torchvision import datasets,transforms
import torch.nn as nn
from torch.optim import SGD
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
import time
from pgd import PGD
from tqdm import tqdm
from dknn import DKNN
from aise import AISE
from collections import deque
#    from models.VGG import VGG

ROOT = "./datasets"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainset = datasets.CIFAR10(root=ROOT,train=True,transform=transforms.ToTensor())
trainloader = DataLoader(trainset,shuffle=True,batch_size=100)

testset = datasets.CIFAR10(root=ROOT,train=False,transform=transforms.ToTensor())
testloader = DataLoader(testset,shuffle=False,batch_size=50)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        cfg1 = [64, 64, 'M']
        cfg2 = [128, 128, 'M']
        cfg3 = [256, 256, 256, 'M']
        cfg4 = [512, 512, 512, 'M']
        cfg5 = [512, 512, 512, 'M']
        self.f1 = self._make_layers(cfg1, 3)
        self.f2 = self._make_layers(cfg2, 64)
        self.f3 = self._make_layers(cfg3, 128)
        self.f4 = self._make_layers(cfg4, 256)
        self.f5 = self._make_layers(cfg5, 512)
        self.layer = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        out1 = self.f1(x)
        out2 = self.f2(out1)
        out3 = self.f3(out2)
        out4 = self.f4(out3)
        out45 = self.f5(out4)
        out5 = self.layer(out45)
        out = out5.view(out5.size(0), -1)
        out = self.classifier(out)
        return [out3, out4, out45, out]
    

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x        
        return nn.Sequential(*layers)

model = VGG()

model.load_state_dict(torch.load("./model_weights/cifar_vgg16.pt", map_location=DEVICE)['state_dict'])
model.eval()
model.to(DEVICE)





def build_nn_clfs(
    model,
    x_train,
    hidden_layer=1,
    n_neighbors=10,
    batch_size=1000,
    class_size=1000,
    ind=None,
    device=DEVICE
):
    
    nn_clfs = []
    x_hidden = []
    model.eval()
    with torch.no_grad():
        for k,x in enumerate(x_train):
            x = x[ind[k]]
            xhs = []
            for i in range(0,x.size(0),batch_size):
                xhs.append(model(x[i:i+batch_size].to(device))[hidden_layer].cpu())
            xhs = torch.cat(xhs,dim=0)
            x_hidden.append(xhs)
            nn_clfs.append(NearestNeighbors(n_neighbors=n_neighbors,n_jobs=-1).fit(xhs.flatten(start_dim=1)))
            
    return nn_clfs,x_hidden


def get_nns(
    model,nn_clfs,train_data,train_hidden,x,y,hl=1,input_shape=(3,32,32),device=DEVICE
):
    
    model.eval()
    with torch.no_grad():
        x_hidden = model(x.to(device))[hl].cpu()
    n_neighbors = nn_clfs[0].n_neighbors
    y_class = [y==i for i in range(10)]
    x_class = [x_hidden[yy] for yy in y_class]
    nns = []
    for i,xx in enumerate(x_class):
        if len(xx):
            nn_inds = nn_clfs[i].kneighbors(xx.flatten(start_dim=1),return_distance=False)
            nns.append(train_data[i][torch.LongTensor(nn_inds)])
    nns = torch.cat(nns,dim=0)
    nns_reordered = torch.zeros((x.size(0),n_neighbors,)+input_shape)
    start_ind = 0
    for yy in y_class:
        end_ind = start_ind+yy.sum()
        nns_reordered[yy] = nns[start_ind:end_ind]
        start_ind = end_ind
        
    return nns_reordered.reshape((-1,)+input_shape),x_hidden


def build_neg_clfs(
    model,
    x_train,
    hidden_layer=1,
    n_neighbors=1,
    batch_size=1000,
    class_size=1000,
    ind=None,
    device=DEVICE
):
    
    nn_clfs = []
    x_hidden = []
    model.eval()
    with torch.no_grad():
        for k,x in enumerate(x_train):
            x = x[ind[k]]
            xhs = []
            for i in range(0,x.size(0),batch_size):
                xhs.append(model(x[i:i+batch_size].to(device))[hidden_layer].cpu())
            xhs = torch.cat(xhs,dim=0)
            x_hidden.append(xhs)
            nn_clfs.append(NearestNeighbors(n_neighbors=n_neighbors,\
                                            n_jobs=-1).fit(xhs.flatten(start_dim=1)))
            
    return nn_clfs,x_hidden


def get_negs(
    model,
    nn_clfs,
    train_data,
    train_hidden,
    x,
    y,
    hl=1,
    input_shape=(3,32,32),
    device=DEVICE,
    targeted=False
):
    
    model.eval()
    with torch.no_grad():
        x_hidden = model(x.to(device))[hl].cpu()
    n_neighbors = nn_clfs[0].n_neighbors
    y_class = [y==i for i in range(10)]
    x_class = [x_hidden[yy] for yy in y_class]
    nns = []
    for i,xx in enumerate(x_class):
        if len(xx):
            if targeted:
                neib_col = []
                dist_col = []
                for j in range(10):
                    if j != i:
                        dists,nn_inds = nn_clfs[j].kneighbors(xx.flatten(start_dim=1),return_distance=True)
                        neib_col.append(train_data[j][torch.LongTensor(nn_inds)])
                        dist_col.append(np.mean(dists,axis=1))
                dists = torch.LongTensor(np.stack(dist_col,axis=1))
                nns.append(torch.stack(neib_col,dim=1)[range(len(xx)),dists.argmin(dim=-1),:,:,:,:])
            else:
                neib_col = []
                for j in range(10):
                    if j != i:
                        neib_col.append(train_data[j][torch.LongTensor(
                            nn_clfs[j].kneighbors(xx.flatten(start_dim=1),return_distance=False)
                            )])
                nns.append(torch.cat(neib_col,dim=1))
    nns = torch.cat(nns,dim=0)
    if targeted:
        nns_reordered = torch.zeros((x.size(0),n_neighbors,)+input_shape)                   
    else:
        nns_reordered = torch.zeros((x.size(0),n_neighbors*9,)+input_shape)
    start_ind = 0
    for yy in y_class:
        end_ind = start_ind+yy.sum()
        nns_reordered[yy] = nns[start_ind:end_ind]
        start_ind = end_ind
    #print(nns.shape)
    return nns_reordered.reshape((-1,)+input_shape),x_hidden


def calc_affinity(nns,x):
    
    s1, s2, s3, s4, s5 = nns.size()
    x = x.unsqueeze(1).repeat_interleave(s2,dim=1)
#     x = x.repeat_interleave(s2,dim=0).reshape((s1, s2, s3, s4, s5))
    
    return (nns-x).pow(2).sum(dim=(1,2,3,4)).sqrt() / s2


def KnnAttack(
    inp, y_inp, nbd, model, targeted=False, 
    x_ot=None, rl=True, eps=8/255, step=2/255, 
    it=10, lamb = 10, layer=1, ch=False, DEVICE=DEVICE
):
    
    choice = ch # set to False if using the softmax-format version
    
    model.eval()
    eta = torch.FloatTensor(*inp.shape).uniform_(-eps, eps)
    inp = inp.to(DEVICE)
    eta = eta.to(DEVICE)
    eta.requires_grad = True
    inp.requires_grad = True
    tau = 1
    
    
    if type(layer) is list:
        neibor_sc_rep = []
        neibor_dc_rep = []
        for j, l in enumerate(layer):
            shape = model(inp[:2].to(DEVICE))[l].size()
            s1 = shape[1]
            s2 = shape[2]
            s3 = shape[3]
            d1 = inp.size(0)
            d2 = nbd[j].size(0) // d1
            
            neibor_sc_rep.append(model(nbd[j].to(DEVICE))[l].reshape((d1,d2,s1,s2,s3)))
            if not rl:
                d2_cl = x_ot[j].size(0) // d1
                neibor_dc_rep.append(model(x_ot[j].to(DEVICE))[l].reshape((d1,d2_cl,s1,s2,s3)))
                # only used for choice False of the untargeted attack
                if not choice:
                    size_cl = d2_cl // 9
    
    else:
    
        shape = model(inp[:2].to(DEVICE))[layer].size()
        s1 = shape[1]
        s2 = shape[2]
        s3 = shape[3]
        d1 = inp.size(0)
        d2 = nbd.size(0) // d1
        
    
        
        neibor_sc_rep = model(nbd.to(DEVICE))[layer].reshape((d1,d2,s1,s2,s3))
        if not rl:
            d2_cl = x_ot.size(0) // d1
            neibor_dc_rep = model(x_ot.to(DEVICE))[layer].reshape((d1,d2_cl,s1,s2,s3))
            # only used for choice False of the untargeted attack
            if not choice:
                size_cl = d2_cl // 9

    for i in range(it):
        
        inpadv = inp + eta
        
        
        if type(layer) is list:
            aff_sum = 0
            for q, l in enumerate(layer):
                if l == 1:
                    tau = 0.1
                else:
                    tau = 1
            
                affinity = calc_affinity(neibor_sc_rep[q],model(inpadv.to(DEVICE))[l])
            
                if rl:
                    affinity = affinity
                else:
                    affinity = - affinity
                    if choice or targeted:
                        negaff = - calc_affinity(neibor_dc_rep[q],model(inpadv.to(DEVICE))[l])
                        affinity = - torch.log(torch.exp(affinity / tau)/(torch.exp(affinity / tau)+torch.exp(negaff / tau)))
                    # only used for choice 1 of the untargeted attack
                    else:
                        negaff = 0
                        for j in range(9):
                            tempaff = - calc_affinity(
                                neibor_dc_rep[q][:,j*size_cl:(j+1)*size_cl],model(inpadv.to(DEVICE))[l]
                            )
                            negaff += torch.exp(tempaff / tau)
                        affinity = - torch.log(torch.exp(affinity / tau)/(torch.exp(affinity / tau)+negaff)).mean()
                    
                affinity = - affinity.mean()
                aff_sum += affinity
            affinity = aff_sum
            
            
        else:

            affinity = calc_affinity(neibor_sc_rep,model(inpadv.to(DEVICE))[layer])
            
            if rl:
                affinity = affinity
            else:
                affinity = - affinity
                if choice or targeted:
                    negaff = - calc_affinity(neibor_dc_rep,model(inpadv.to(DEVICE))[layer])
                    affinity = - torch.log(torch.exp(affinity / tau)/(torch.exp(affinity / tau)+torch.exp(negaff / tau)))
                # only used for choice 1 of the untargeted attack
                else:
                    negaff = 0
                    for j in range(9):
                        tempaff = - calc_affinity(
                            neibor_dc_rep[:,j*size_cl:(j+1)*size_cl],model(inpadv.to(DEVICE))[layer]
                        )
                        negaff += torch.exp(tempaff / tau)
                    affinity = - torch.log(torch.exp(affinity / tau)/(torch.exp(affinity / tau)+negaff)).mean()
                
            affinity = - affinity.mean()
        pred_adv = model(inpadv)[-1]
        loss = lamb*affinity
        grad_sign = torch.autograd.grad(loss, inpadv, only_inputs=True,\
                                        retain_graph = False)[0].sign()
        # affinity.backward()
        pert = step * grad_sign
        inpadv = (inpadv-pert).clamp(0.0,1.0)
        tempeta = (inpadv - inp).clamp(-eps, eps)
        eta = tempeta
        
    return inp+eta
    


    

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
    




y_train = np.array(trainset.targets)
train_data = [torch.FloatTensor(trainset.data[y_train==i].transpose(0,3,1,2)/255.) for i in range(10)]


def test_attack(
    y_train,
    train_data,
    relax=False, # use relaxation or not; In the attack, we will mainly consider relax=False
    choice = False, # use which version of contrastive loss, set to False if using the softmax-format version
    targeted = False, # whether or not to use targeted attacks
    loss_fn = nn.CrossEntropyLoss(),
    pgd = PGD(eps=8/255.,step=2/255.,max_iter=10),
    nn_t_class = 6, # k for the true class
    nn_f_class = 5, # k for the other classes
    count_up_knn = 10, # max number of batches for evaluation
    layers = [0,1], # the selected layers, choices are 0, 1, 2, 3; 0/1/2 is the third/fourth/fifth layer, 3 is the output
    class_samp_size = 1000, #number of samples per class for constructing knn structure
    random_state = 1234
):

    x_neg = None  
    nn_clfs = None

    test_correct_rob = 0
    test_correct = 0
    test_correct_knn = 0
    test_correct_knnrob = 0
    test_correct_knnadvdnn = 0
    test_correct_knnpgd = 0
    
    test_correct_rails = 0
    test_correct_railsadv = 0
    test_correct_railsknnadv = 0
    
    test_total = 0
    test_total_knn = 0
    count_knn_eval = 0
    
    
    np.random.seed(random_state)
    ind_samp =  [np.random.choice(np.arange(5000),size=class_samp_size,replace=False) for i in range(10)]
    ind_samp_data = [np.where(y_train==i)[0][ind_samp[i]] for i in range(10)]
    ind_samp_flat = [x for sublist in ind_samp_data for x in sublist]
    train_samp = torch.FloatTensor(trainset.data.transpose(0,3,1,2)/255.)[ind_samp_flat]
    y_samp = torch.LongTensor(y_train)[ind_samp_flat]
    
    if type(layers) is list:
    
        CIFAR_CONFIGS = {
            "start_layer": 1,
            "n_class": 10,
            "aise_params": [
                {"hidden_layer": 2, "sampling_temperature": 1, "max_generation": 10, "mut_range": (.005, .015)},
                {"hidden_layer": 3, "sampling_temperature": 10, "max_generation": 5, "mut_range": (.005, .015)}
            ]
        }
        
        hidden_l = [layers[0]+2,layers[1]+2]
    
    elif layers == 1:
        
        CIFAR_CONFIGS = {
            "start_layer": 1,
            "n_class": 10,
            "aise_params": [
                #{"hidden_layer": 2, "sampling_temperature": 1, "max_generation": 10, "mut_range": (.005, .015)},
                {"hidden_layer": 3, "sampling_temperature": 10, "max_generation": 5, "mut_range": (.005, .015)}
            ]
        }
        hidden_l = [layers+2]
    
    else:
        
        CIFAR_CONFIGS = {
            "start_layer": 1,
            "n_class": 10,
            "aise_params": [
                {"hidden_layer": 2, "sampling_temperature": 1, "max_generation": 10, "mut_range": (.005, .015)}
            ]
        }
        hidden_l = [layers+2]
    
    model_rails = VGG()
    model_rails.load_state_dict(torch.load(
        "./model_weights/cifar_vgg16.pt", map_location=DEVICE
    )['state_dict'])
    model_rails.eval()
    model_rails.to(DEVICE)
    

    model.eval()
    dknn = DKNN(model, train_samp, y_samp, hidden_layers=hidden_l, device=DEVICE)
    
    rails = RAILS(model_rails, CIFAR_CONFIGS, train_samp, y_samp)


    train_data_samp = [torch.FloatTensor(trainset.data[ind_samp_data[i]].transpose(0,3,1,2)/255.) for i in range(10)]

    
    if type(layers) is list:
        nn_clfs = []
        train_hidden = []
        neg_clfs = []
        neg_hidden = []
        n_l = len(layers)
        for layer in layers:
            nn_clfs_tep, train_hidden_tep = build_nn_clfs(
                model,train_data,hidden_layer=layer,n_neighbors=nn_t_class,class_size=class_samp_size,ind=ind_samp
            )
        
            if not relax:
                neg_clfs_tep, neg_hidden_tep = build_neg_clfs(
                    model,train_data,hidden_layer=layer,n_neighbors=nn_f_class,class_size=class_samp_size,ind=ind_samp
                )
            
            nn_clfs.append(nn_clfs_tep)
            train_hidden.append(train_hidden_tep)
            if not relax:
                neg_clfs.append(neg_clfs_tep)
                neg_hidden.append(neg_hidden_tep)
        
        
    else:
    
        nn_clfs, train_hidden = build_nn_clfs(
            model,train_data,hidden_layer=layers,n_neighbors=nn_t_class,class_size=class_samp_size,ind=ind_samp
        )
    
        if not relax:
            neg_clfs, neg_hidden = build_neg_clfs(
                model,train_data,hidden_layer=layers,n_neighbors=nn_f_class,class_size=class_samp_size,ind=ind_samp
            )

    for x,y in testloader:
        count_knn_eval = count_knn_eval + 1

        if count_knn_eval < count_up_knn:
            x_adv = pgd.generate(model,x,y,device=DEVICE)
            if type(layers) is list:
                x_mem = []
                x_neg = []
                for i, layer in enumerate(layers):
                    x_mem_tep, _ = get_nns(model,nn_clfs[i],train_data_samp,train_hidden[i],x,y,hl=layer)
                    x_mem.append(x_mem_tep)
                    if not relax:
                        x_neg_tep, _ = get_negs(model,neg_clfs[i],train_data_samp,neg_hidden[i],x,y,hl=layer,targeted=targeted)
                        x_neg.append(x_neg_tep)
                    
            else:
                x_mem, _ = get_nns(model,nn_clfs,train_data_samp,train_hidden,x,y,hl=layers)
                if not relax:
                    x_neg, _ = get_negs(model,neg_clfs,train_data_samp,neg_hidden,x,y,hl=layers,targeted=targeted)

            x_knnadv = KnnAttack(
                x, y, x_mem, model, targeted=targeted, 
                x_ot = x_neg, rl = relax, 
                eps=8/255, step=2/255, it=20, lamb = 1000, layer=layers, ch=choice, DEVICE=DEVICE
            )

        model.eval()

        with torch.no_grad():

            if count_knn_eval < count_up_knn:
                pred = model(x.to(DEVICE))[-1].max(dim=1)[1]
                test_correct += (pred==y.to(DEVICE)).sum().item()
                pred_adv = model(x_adv.to(DEVICE))[-1].max(dim=1)[1]
                test_correct_rob += (pred_adv==y.to(DEVICE)).sum().item()
                test_total += x.size(0)

                # knn attack acc
                pred_dknn = dknn(x.to(DEVICE)).argmax(axis=1)
                pred_knnadv = dknn(x_knnadv.to(DEVICE)).argmax(axis=1)
                pred_knnadvdnn = model(x_knnadv.to(DEVICE))[-1].max(dim=1)[1]
                pred_knnpgdadv = dknn(x_adv.to(DEVICE)).argmax(axis=1)
                
                pred_rails = rails.predict(x.to(DEVICE)).argmax(axis=1)
                pred_railsadv = rails.predict(x_adv.to(DEVICE)).argmax(axis=1)
                pred_railsknnadv = rails.predict(x_knnadv.to(DEVICE)).argmax(axis=1)
                
                test_correct_knn += (pred_dknn==y.numpy()).astype("float").sum()
                test_correct_knnadvdnn += (pred_knnadvdnn==y.to(DEVICE)).sum().item()
                test_correct_knnrob += (pred_knnadv==y.numpy()).astype("float").sum()
                test_correct_knnpgd += (pred_knnpgdadv==y.numpy()).astype("float").sum()
                
                test_correct_rails += (pred_rails==y.numpy()).astype("float").sum()
                test_correct_railsadv += (pred_railsadv==y.numpy()).astype("float").sum()
                test_correct_railsknnadv += (pred_railsknnadv==y.numpy()).astype("float").sum()
                
                test_total_knn += x.size(0)

    print({
        "test_acc": test_correct/test_total,
        "test_acc_rob": test_correct_rob/test_total,
        "test_acc_knn": test_correct_knn/test_total_knn,
        "test_acc_knnrob": test_correct_knnrob/test_total_knn,
        "test_acc_knnadvdnn": test_correct_knnadvdnn/test_total_knn,
        "test_acc_knnpgd": test_correct_knnpgd/test_total_knn,
        "test_acc_rails": test_correct_rails/test_total_knn,
        "test_acc_railsadv": test_correct_railsadv/test_total_knn,
        "test_acc_railsknnadv": test_correct_railsknnadv/test_total_knn
        
    })
    
test_attack(
    y_train,
    train_data,
    targeted = False, # whether or not to use targeted attacks
    random_state = 3#135
)