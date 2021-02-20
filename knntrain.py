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

ROOT = "./datasets"

trainset = datasets.CIFAR10(root=ROOT,train=True,transform=transforms.ToTensor())
trainloader = DataLoader(trainset,shuffle=True,batch_size=125)

testset = datasets.CIFAR10(root=ROOT,train=False,transform=transforms.ToTensor())
testloader = DataLoader(testset,shuffle=True,batch_size=50)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = VGG()
import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        #self.features = self._make_layers(cfg[vgg_name])
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
        #self.classifier = nn.Linear(512, 10)
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
model.to(DEVICE)


# model.load_state_dict(torch.load(
#     "./model_weights/cifar_vgg16.pt", map_location=DEVICE
# )['state_dict'])
# model.eval()
# model.to(DEVICE)



def build_nn_clfs(model,x_train,hidden_layer=1,n_neighbors=10,\
                  batch_size=1000,class_size=1000,ind=None,device=DEVICE):
    nn_clfs = []
    x_hidden = []
    model.eval()
    with torch.no_grad():
        for k,x in enumerate(x_train):
            #x = x[np.random.choice(np.arange(x.size(0)),size=class_size,replace=False)]
            x = x[ind[k]]
            xhs = []
            for i in range(0,x.size(0),batch_size):
                xhs.append(model(x[i:i+batch_size].to(device))[hidden_layer].cpu())
            xhs = torch.cat(xhs,dim=0)
            x_hidden.append(xhs)
            nn_clfs.append(NearestNeighbors(n_neighbors=n_neighbors,\
                                            n_jobs=-1).fit(xhs.flatten(start_dim=1)))
    return nn_clfs,x_hidden


def get_nns(model,nn_clfs,train_data,train_hidden,x,y,hl=1,\
            input_shape=(3,32,32),device=DEVICE):
    model.eval()
    with torch.no_grad():
        x_hidden = model(x.to(device))[hl].cpu()
    n_neighbors = nn_clfs[0].n_neighbors
    y_class = [y==i for i in range(10)]
    x_class = [x_hidden[yy] for yy in y_class]
    nns = []
    for i,xx in enumerate(x_class):
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



def build_neg_clfs(model,x_train,hidden_layer=1,n_neighbors=1,\
                  batch_size=1000,class_size=1000,ind=None,device=DEVICE):
    nn_clfs = []
    x_hidden = []
    model.eval()
    with torch.no_grad():
        for k,x in enumerate(x_train):
            #x = x[np.random.choice(np.arange(x.size(0)),size=class_size,replace=False)]
            x = x[ind[k]]
            xhs = []
            for i in range(0,x.size(0),batch_size):
                xhs.append(model(x[i:i+batch_size].to(device))[hidden_layer].cpu())
            xhs = torch.cat(xhs,dim=0)
            x_hidden.append(xhs)
            nn_clfs.append(NearestNeighbors(n_neighbors=n_neighbors,\
                                            n_jobs=-1).fit(xhs.flatten(start_dim=1)))
    return nn_clfs,x_hidden


def get_negs(model,nn_clfs,train_data,train_hidden,x,y,hl=1,\
            input_shape=(3,32,32),device=DEVICE):
    model.eval()
    with torch.no_grad():
        x_hidden = model(x.to(device))[hl].cpu()
    n_neighbors = nn_clfs[0].n_neighbors*9
    y_class = [y==i for i in range(10)]
    x_class = [x_hidden[yy] for yy in y_class]
    nns = []
    for i,xx in enumerate(x_class):
        for j in range(10):
            if j != i:
                nn_inds = nn_clfs[j].kneighbors(xx.flatten(start_dim=1),\
                                                return_distance=False)
                if (i == 0 and j == 1) or (i > 0 and j == 0):
                    neib_col = train_data[j][torch.LongTensor(nn_inds)]
                else:
                    neib_col = torch.cat((neib_col,\
                                          train_data[j][torch.LongTensor(nn_inds)]),1)
        nns.append(neib_col)
    nns = torch.cat(nns,dim=0)
    nns_reordered = torch.zeros((x.size(0),n_neighbors,)+input_shape)
    start_ind = 0
    for yy in y_class:
        end_ind = start_ind+yy.sum()
        nns_reordered[yy] = nns[start_ind:end_ind]
        start_ind = end_ind
    return nns_reordered.reshape((-1,)+input_shape),x_hidden



def calc_affinity(nns,x):
    #nns - (batchsize, nearest neighbor size, ch1, ch2, ch3)
    s1, s2, s3, s4, s5 = nns.size()
    x = x.repeat_interleave(s2,dim=0).reshape((s1, s2, s3, s4, s5))
    return (nns-x).pow(2).sum(dim=(1,2,3,4)).sqrt() / s2#.mean()

def KnnAttack(inp, y_inp, nbd, model, x_ot=None, rl=True,\
              eps=8/255, step=2/255, it=10, lamb = 10, layer=1, ch=False, DEVICE=DEVICE):
    choice = ch #set to False if using the softmax-format version
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    eta = torch.FloatTensor(*inp.shape).uniform_(-eps, eps)
    inp = inp.to(DEVICE)
    eta = eta.to(DEVICE)
    eta.requires_grad = True
    inp.requires_grad = True
    tau = 1
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
        #only used for choice False of the untargeted attack
        if not choice:
            size_cl = d2_cl // 9
                #ind.append([k for i in range(d1) for k in range(j*size_cl+d2_cl*i,(j+1)*size_cl+d2_cl*i)])
        ###
        
    
    
    for i in range(it):
        inpadv = inp + eta

        affinity = calc_affinity(neibor_sc_rep,model(inpadv.to(DEVICE))[layer])
        
        if rl:
            affinity = affinity
        else:
            affinity = - affinity
            if choice:
                negaff = - calc_affinity(neibor_dc_rep,model(inpadv.to(DEVICE))[layer])
                affinity = - torch.log(torch.exp(affinity / tau) /\
                                       (torch.exp(affinity / tau)+torch.exp(negaff / tau)))
            #only used for choice 1 of the untargeted attack
            else:
                negaff = 0
                for j in range(9):
                    tempaff = - calc_affinity(neibor_dc_rep[:,j*size_cl:(j+1)*size_cl],\
                                              model(inpadv.to(DEVICE))[layer])
                    negaff += torch.exp(tempaff / tau)
                affinity = - torch.log(torch.exp(affinity / tau) /\
                                   (torch.exp(affinity / tau)+negaff))
            #######
            
        affinity = - affinity.mean()
        pred_adv = model(inpadv)[-1]
        loss_ce = - loss_fn(pred_adv, y_inp.to(DEVICE))
        loss = loss_ce + lamb*affinity
        grad_sign = torch.autograd.grad(loss, inpadv, only_inputs=True,\
                                        retain_graph = False)[0].sign()
        #affinity.backward()
        pert = step * grad_sign
        inpadv = (inpadv-pert).clamp(0.0,1.0)
        tempeta = (inpadv - inp).clamp(-eps, eps)
        eta = tempeta
    return inp+eta



def loss_aff(inp, nbd, nbd_neg=None, rl=True, layer=1, ch=False, DEVICE=DEVICE):
        
    choice = ch #set to False if using the softmax-format version
    tau = 1
    shape = model(inp[:2].to(DEVICE))[layer].size()
    s1 = shape[1]
    s2 = shape[2]
    s3 = shape[3]
    d1 = inp.size(0)
    d2 = nbd.size(0) // d1
    neibor_sc_rep = model(nbd.to(DEVICE))[layer].reshape((d1,d2,s1,s2,s3))
    if not rl:
        d2_cl = nbd_neg.size(0) // d1
        neibor_dc_rep = model(nbd_neg.to(DEVICE))[layer].reshape((d1,d2_cl,s1,s2,s3))
        #only used for choice False of the untargeted attack
        if not choice:
            size_cl = d2_cl // 9


    affinity = calc_affinity(neibor_sc_rep,model(inp.to(DEVICE))[layer])

    if rl:
        affinity = affinity
    else:
        affinity = - affinity
        if choice:
            negaff = - calc_affinity(neibor_dc_rep,model(inp.to(DEVICE))[layer])
            affinity = - torch.log(torch.exp(affinity / tau) /\
                                   (torch.exp(affinity / tau)+torch.exp(negaff / tau)))
        #only used for choice 1 of the untargeted attack
        else:
            negaff = 0
            for j in range(9):
                tempaff = - calc_affinity(neibor_dc_rep[:,j*size_cl:(j+1)*size_cl],\
                                          model(inp.to(DEVICE))[layer])
                negaff += torch.exp(tempaff / tau)
            affinity = - torch.log(torch.exp(affinity / tau) /\
                               (torch.exp(affinity / tau)+negaff))
        #######

    affinity = affinity.mean()
    return affinity


y_train = np.array(trainset.targets)
train_data = [torch.FloatTensor(trainset.data[y_train==i].transpose(0,3,1,2)/255.) for i in range(10)]


BURN_IN = 3
EPS = 0.02
#use relaxation or not; In the attack, we will mainly consider relax=False
relax = False
x_neg = None

#use which version of contrastive loss, set to False if using the softmax-format version
choice = False

#the selected layers, choices are 0, 1, 2, 3; 0/1/2 is the third/fourth/fifth layer, 3 is the output
layers = 2

loss_fn = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(),lr=1e-3,momentum=0.9,weight_decay=1e-4,nesterov=True)
pgd = PGD(eps=8/255.,step=2/255.,max_iter=10)
# scheduler = lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.1)
EPOCHS = 30
nn_clfs = None
lt1 = 0#0.01  #penalty on knn loss
lt2 = 0#100
class_samp_size = 1000 #number of samples per class for constructing knn structure
nn_t_class = 5
nn_f_class = 3


for ep in range(EPOCHS):

    
    train_loss = 0.
    train_correct = 0.
    train_total = 0.
    train_clean_correct = 0.

    with tqdm(trainloader,desc=f"{ep+1}/{EPOCHS} epochs:") as t:
        for i,(x,y) in enumerate(t):
            if ep >= BURN_IN and ep < BURN_IN+2:
                lt1 = 0.1
                lt2 = 1000
            else:
                lt1 = 1
                lt2 = 100
            #model.train()
            model.eval()
            if i % 10 == 0 and ep >= BURN_IN:
#                 ind_samp_tr = [np.where(y_train==i)[0][np.random.choice(\
#                                      np.arange(5000),size=class_samp_size,replace=False)] for i in range(10)]
                #ind_samp_tr_flat = [x for sublist in ind_samp_tr for x in sublist]
                
                ind_samp_tr = [np.random.choice(\
                                     np.arange(5000),size=class_samp_size,replace=False) for i in range(10)]
            
                ind_samp_data_tr = [np.where(y_train==i)[0][ind_samp_tr[i]] for i in range(10)]
                
                nn_clfs, train_hidden = build_nn_clfs(model,train_data,hidden_layer=layers,\
                                              n_neighbors=nn_t_class,class_size=class_samp_size,ind=ind_samp_tr)
                if not relax:
                    neg_clfs, neg_hidden = build_neg_clfs(model,train_data,hidden_layer=layers,\
                                              n_neighbors=nn_f_class,class_size=class_samp_size,ind=ind_samp_tr)
                    
                data_samp = [torch.FloatTensor(trainset.data[ind_samp_data_tr[i]].transpose(0,3,1,2)/255.)\
                                   for i in range(10)]
            
            
            if nn_clfs is not None:
                x_mem, _ = get_nns(model,nn_clfs,data_samp,train_hidden,x,y,hl=layers)
                if not relax:
                    x_neg, _ = get_negs(model,neg_clfs,data_samp,neg_hidden,x,y,hl=layers)
                x_adv = KnnAttack(x, y, x_mem, model, x_ot = x_neg, rl = relax, eps=4/255,\
                                  step=2/255,it=10, lamb = lt2, layer=layers, ch=choice, DEVICE=DEVICE)
                
                
                
                model.train()
                *_,out = model(x_adv.detach().to(DEVICE))
                
                *_,out_clean = model(x.detach().to(DEVICE))
                pred_clean = out_clean.max(1)[1].detach().cpu()
                train_clean_correct += (pred_clean==y).sum().item()
                
                loss_ce = loss_fn(out,y.to(DEVICE))
                aff = loss_aff(x_adv, x_mem, nbd_neg=x_neg, rl=relax, layer=layers, ch=choice, DEVICE=DEVICE)

                loss = loss_ce + lt1*aff
                train_loss += loss.item()
                pred = out.max(1)[1].detach().cpu()
                train_correct += (pred==y).sum().item()
                train_total += x.size(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.set_postfix({
                    "train_loss": train_loss/train_total,
                    "train_clean_acc": train_clean_correct/train_total,
                    "train_acc": train_correct/train_total
                })
            else:
                model.train()
                *_,out = model(x.to(DEVICE))
                loss = loss_fn(out,y.to(DEVICE))
                train_loss += loss.item()*x.size(0)
                pred = out.max(dim=1)[1].detach().cpu()
                train_correct += (pred==y).sum().item()
                train_total += x.size(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.set_postfix({
                    "train_loss": train_loss/train_total,
                    "train_acc": train_correct/train_total
                })
            if i == len(trainloader)-1 and ep >=BURN_IN:
                count_up_knn = 10
                test_correct_rob = 0
                test_correct = 0
                test_correct_knn = 0
                test_correct_knnrob = 0
                test_correct_knnpgd = 0
                test_total = 0
                test_total_knn = 0
                count_knn_eval = 0
                
#                 index = np.random.choice(np.arange(50000),\
#                          size=1000,replace=False)
                

#                 ind_samp =  [np.where(y_train==i)[0][np.random.choice(\
#                                      np.arange(5000),size=class_samp_size,replace=False)] for i in range(10)]
        
                ind_samp =  [np.random.choice(\
                                     np.arange(5000),size=class_samp_size,replace=False) for i in range(10)]
                ind_samp_data = [np.where(y_train==i)[0][ind_samp[i]] for i in range(10)]
                ind_samp_flat = [x for sublist in ind_samp_data for x in sublist]
                train_samp = torch.FloatTensor(trainset.data.\
                                               transpose(0,3,1,2)/255.)[ind_samp_flat]
                y_samp = torch.LongTensor(y_train)[ind_samp_flat]
                
#                 train_samp = torch.FloatTensor(trainset.data.\
#                                                transpose(0,3,1,2)/255.)[index]
#                 y_samp = torch.LongTensor(y_train)[index]
                model.eval()
                dknn = DKNN(model, train_samp, y_samp,\
                            hidden_layers=[layers+2], device=DEVICE)
                    


                train_data_samp = [torch.FloatTensor(trainset.data[ind_samp_data[i]].transpose(0,3,1,2)/255.)\
                                   for i in range(10)]
                nn_clfs, train_hidden = build_nn_clfs(model,train_data,hidden_layer=layers,\
                                                n_neighbors=nn_t_class,class_size=class_samp_size,ind=ind_samp)

                if not relax:
                    neg_clfs, neg_hidden = build_neg_clfs(model,train_data,hidden_layer=layers,\
                                                    n_neighbors=nn_f_class,class_size=class_samp_size,ind=ind_samp)

                    
#                 data_samp = [torch.FloatTensor(trainset.data[ind_samp_tr[i]].transpose(0,3,1,2)/255.)\
#                                    for i in range(10)]

                for x,y in testloader:
                    count_knn_eval = count_knn_eval + 1
                    x_adv = pgd.generate(model,x,y,device=DEVICE)
                    #knn attack (informal)
#                     if count_knn_eval < count_up_knn:
#                         x_mem, _ = get_nns(model,nn_clfs,train_data_samp,train_hidden,x,y)
#                         if not relax:
#                             x_neg, _ = get_negs(model,neg_clfs,train_data_samp,neg_hidden,x,y)
#                         x_knnadv = KnnAttack(x, y, x_mem, model, x_ot = x_neg, rl = relax, eps=8/255,\
#                                       step=2/255,it=10, lamb = 1000, DEVICE=DEVICE)
                    

            
            
#                     x_knnadv = dknnatt.generate(x, y)
    
    
    ####
                    model.eval()
                    with torch.no_grad():
                        pred = model(x.to(DEVICE))[-1].max(dim=1)[1]
                        test_correct += (pred==y.to(DEVICE)).sum().item()
                        pred_adv = model(x_adv.to(DEVICE))[-1].max(dim=1)[1]
                        test_correct_rob += (pred_adv==y.to(DEVICE)).sum().item()
                        test_total += x.size(0)
                        
                        if count_knn_eval < count_up_knn:
                        #knn attack acc
                            pred_dknn = dknn(x.to(DEVICE)).argmax(axis=1)
                            pred_knnadv = dknn(x_adv.to(DEVICE)).argmax(axis=1)
                            pred_knnpgdadv = dknn(x_adv.to(DEVICE)).argmax(axis=1)
#                         #pred_knnadv = model(x_knnadv.to(DEVICE))[-1].max(dim=1)[1]
                            test_correct_knn += (pred_dknn==y.numpy()).astype("float").sum()
                            test_correct_knnrob += (pred_knnadv==y.numpy()).astype("float").sum()
                            test_correct_knnpgd += (pred_knnpgdadv==y.numpy()).astype("float").sum()
                            test_total_knn += x.size(0)
                        #
                t.set_postfix({
                    "train_loss": train_loss/train_total,
                    "train_acc": train_correct/train_total,
                    "test_acc": test_correct/test_total,
                    "test_acc_rob": test_correct_rob/test_total,
                    "test_acc_knn": test_correct_knn/test_total_knn,
                    "test_acc_knnrob": test_correct_knnrob/test_total_knn,
                    "test_acc_knnpgd": test_correct_knnpgd/test_total_knn
                })
                state = {
                        'state_dict': model.state_dict()
                    }
                torch.save(state, 'models/relaxtest0d1and1000lay5.pt')
                print('saved')
#     scheduler.step()