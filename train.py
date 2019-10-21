import torch 
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from numpy import random as ra
from os import  path
from sklearn.metrics import  roc_auc_score 
import model.DLRM_Net.DLRM_Net.
from torch.utils.data.sampler import  WeightedRandomSampler
import data.collate_wrapper.collate_wrapper

wei_data=np.load('all_train.npz')
wei_data_y=wei_data['y']
weights=[1000  if target==1 else 6  for target in wei_data_y]
sampler = WeightedRandomSampler(weights,num_samples=len(wei_data_y)*2, replacement=True)
#train_data包含稠密特征的维度，和稀疏特征的个数
train_data = CriteoDataset(
    dataset="all_train.npz",
    randomize="total",
    split="none",
    raw_path="",
    pro_data="",
)
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=1000,
    #shuffle=False,
    sampler=sampler,
    num_workers=0,
    collate_fn=collate_wrapper,#该函数的输入是一个batch的数据的list格式
    pin_memory=False,
    drop_last=False)
#train_data包含稠密特征的维度，和稀疏特征的个数
test_data = CriteoDataset(
    dataset="all_eval.npz",
    randomize="total",
    split="none",
    raw_path="",
    pro_data="",
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=100000, ##test_data.dataset_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_wrapper,#该函数的输入是一个batch的数据的list格式
    pin_memory=False,
    drop_last=False)


arch_mlp_bot="4-1024-1024-512-256-64"
arch_mlp_top="1024-1024-512-256-1"
ln_bot = np.fromstring(arch_mlp_bot, dtype=int, sep="-")
ln_emb = train_data.counts
m_den = train_data.m_den
ln_bot[0] = m_den
m_spa = 64 # 隐藏特征大小
arch_interaction_op="cat"
num_fea = ln_emb.size + 1  # num sparse + num dense features #=8
m_den_out = ln_bot[ln_bot.size - 1]#=2
arch_interaction_itself=False
arch_interaction_itself=False
sync_dense_params=True
loss_threshold=0.0
loss_function='bce'
inference_only=False

if arch_interaction_op == "dot": #是这个
    # approach 1: all
    # num_int = num_fea * num_fea + m_den_out
    # approach 2: unique
    if arch_interaction_itself: #=False
        num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
    else:
        num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out#=8
elif arch_interaction_op == "cat":
    num_int = num_fea * m_den_out
else:
    sys.exit(
        "ERROR: --arch-interaction-op="
        + arch_interaction_op
        + " is not supported"
    )
arch_mlp_top_adjusted = str(num_int) + "-" + arch_mlp_top
ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")



k=0
nepochs=30
numpy_rand_seed=2019
learning_rate=0.001

#device = torch.device("cuda:0")

dlrm = DLRM_Net(
    m_spa,
    ln_emb,
    ln_bot,
    ln_top,
    arch_interaction_op=arch_interaction_op,
    arch_interaction_itself=arch_interaction_itself,
    sigmoid_bot=-1,
    sigmoid_top=ln_top.size - 2,
    sync_dense_params=sync_dense_params, #sync_dense_params=True
    loss_threshold=loss_threshold,
    ndevices=-1
)
dlrm=dlrm.cuda(1)
# dlrm=dlrm.to(device)
# if torch.cuda.device_count() > 1:
#     dlrm = nn.DataParallel(dlrm)

# print(dlrm)
ops=torch.optim.SGD(dlrm.parameters(),learning_rate)

def dlrm_wrap(X, lS_o, lS_i):
        return dlrm(X, lS_o, lS_i)
    
if loss_function == "mse":
    loss_fn = torch.nn.MSELoss(reduction="mean")
elif loss_function == "bce":
    loss_fn = torch.nn.BCELoss(reduction="mean")
else:
    sys.exit("ERROR: --loss-function=" + args.loss_function + " is not supported")

while k < nepochs:
    k=k+1
    '''测试部分'''
    for (X, lS_o, lS_i,T) in test_loader:
        X=X.cuda(1)
        lS_o=[S_o.cuda(1) for S_o in lS_o]
        lS_i= [S_i.cuda(1) for S_i in lS_i]
        
        Z = dlrm(X, lS_o, lS_i)
        S = Z.detach().cpu().numpy()
        T = T.detach().cpu().numpy()
        auc=roc_auc_score(T,S)
        print("第{}轮测试集AUC : {}".format(k,auc))
    
    for j, (X, lS_o, lS_i, T) in enumerate(train_loader):
        #print("第{}轮循环的第{}batch".format(k,j))
        X=X.cuda(1)
        lS_o=[S_o.cuda(1) for S_o in lS_o]
        lS_i= [S_i.cuda(1) for S_i in lS_i]
        T=T.cuda(1)
        
        Z = dlrm(X, lS_o, lS_i)
        E = loss_fn(Z, T)
       
        #print("loss={}".format(E))
       # L = E.detach().cpu().numpy()  # numpy array
        S = Z.detach().cpu().numpy()  # numpy array
        T = T.detach().cpu().numpy()  # numpy array
        #print("auc : {}".format(auc(T,S)))
        mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
        A = np.sum((np.round(S, 0) == T).astype(np.uint8)) / mbs
        if j%5000==0:
            print("第{}轮循环   的第{}batch loss：{}        train-acc：{}".format(k,j,E,A))
        ops.zero_grad()
        E.backward()
        ops.step()