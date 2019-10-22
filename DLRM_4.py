import torch 
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from numpy import random as ra
from os import  path
from sklearn.metrics import  roc_auc_score 

#将训练数据转化为模型输入数据格式
class CriteoDataset(Dataset):
    def __init__(self, dataset, randomize="total", split="none", raw_path="", pro_data=""):
        days=7
        with np.load(dataset) as data:
            X_int = data["X_int"] # continuous  feature 数据种的必要成分，数值型数据，类别数据，标签，每个类别数据one-hot维度，其中
            X_cat = data["X_cat"] # categorical feature 列别特征的值是每个one-hot数据的位置索引
            y = data["y"]         # target
            self.counts = data["counts"]
        self.m_den = X_int.shape[1] #神经网络输入大小
        self.n_emb = len(self.counts) #emb个数
        self.dataset_size=len(y)
        print("Sparse features = %d, Dense features = %d" % (self.n_emb, self.m_den))
        indices = np.arange(len(y))
        if split == "none":
            # randomize data
            if randomize == "total":
                indices = np.random.permutation(indices) #打乱数据
                print("Randomized indices...")
            self.samples_list = [(X_int[i], X_cat[i], y[i]) for i in indices]
        else:
            indices = np.array_split(indices, days)  #划分数据集
            #print(indices)
            # randomize each day's dataset
            if randomize == "day" or randomize == "total":
                for i in range(len(indices)):
                    indices[i] = np.random.permutation(indices[i]) #打乱数据
            train_indices = np.concatenate(indices[:-1]) #d#连接数据
            #print("train_indices:"+str(train_indices))
            test_indices = indices[-1]
            val_indices, test_indices = np.array_split(test_indices, 2)
            print("Defined %s indices..." % (split))
            # randomize all data in training set
            if randomize == "total":
                train_indices = np.random.permutation(train_indices)
                print("Randomized indices...")
            # create training, validation, and test sets
            if split == 'train':
                self.samples_list = [(X_int[i], X_cat[i], y[i]) for i in train_indices]
            elif split == 'val':
                self.samples_list = [(X_int[i], X_cat[i], y[i]) for i in test_indices]
            elif split == 'test':
                self.samples_list = [(X_int[i], X_cat[i], y[i]) for i in val_indices]
        print("Split data according to indices...")

    def __getitem__(self, index): ##__getitem__()取一条数据的过程
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]
        X_int, X_cat, y = self.samples_list[index]
        X_int, X_cat, y = self._default_preprocess(X_int, X_cat, y)
        return X_int, X_cat, y
    
    def _default_preprocess(self, X_int, X_cat, y):   #
        X_int = torch.log(torch.tensor(X_int, dtype=torch.float) + 1) #将数值数据指数化
        X_cat = torch.tensor(X_cat, dtype=torch.long)   #one-hot的位置索引转化为长整型
        y = torch.tensor(y.astype(np.float32)) #根据损失函数的不同设定不同格式的标签，二分类交叉熵损失需要浮点型数据
        # print("Converted to tensors...done!")
        return X_int, X_cat, y
    def __len__(self):
        return len(self.samples_list)



#该函数生成batch大的数据方便模型处理，
def collate_wrapper(list_of_tuples):
    transposed_data = list(zip(*list_of_tuples))
    X_int = torch.stack(transposed_data[0], 0)
    X_cat = torch.stack(transposed_data[1], 0)
    T       = torch.stack(transposed_data[2], 0).view(-1,1)
    sz0 = X_cat.shape[0]
    sz1 = X_cat.shape[1]
    if 0:
        lS_i = [X_cat[:, i].pin_memory() for i in range(sz1)]
        lS_o = [torch.tensor(range(sz0)).pin_memory() for _ in range(sz1)]
        return X_int.pin_memory(), lS_o, lS_i, T.pin_memory()
    else:
        lS_i = [X_cat[:, i] for i in range(sz1)]
        lS_o = [torch.tensor(range(sz0)) for _ in range(sz1)]
        return X_int, lS_o, lS_i, T


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
    batch_size=50000, ##test_data.dataset_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_wrapper,#该函数的输入是一个batch的数据的list格式
    pin_memory=False,
    drop_last=False)


from torch.utils.data.sampler import  WeightedRandomSampler
wei_data=np.load('all_train.npz')
wei_data_y=wei_data['y']
weights=[1000  if target==1 else 6  for target in wei_data_y]
sampler = WeightedRandomSampler(weights,num_samples=len(wei_data_y), replacement=True)

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
    batch_size=500,
    #shuffle=False,
    sampler=sampler,
    num_workers=0,
    collate_fn=collate_wrapper,#该函数的输入是一个batch的数据的list格式
    pin_memory=False,
    drop_last=False)
class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]
            LL = nn.Linear(int(n), int(m), bias=True)
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
        return torch.nn.Sequential(*layers)  #*表示当作位置参数传进函数

    def create_emb(self, m, ln):
        emb_l = nn.ModuleList()
        for i in range(0, ln.size):
            n = ln[i]
            EE = nn.EmbeddingBag(n, m, mode="sum",sparse=True)
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            # approach 1
            EE.weight.data = torch.tensor(W, requires_grad=True)
            emb_l.append(EE)

        return emb_l

    def __init__(
        self,
        m_spa=None, #稀疏特征隐特征个数
        ln_emb=None, #稀疏特征的大小
        ln_bot=None, # 第一层全连接
        ln_top=None,# 第二层全连接
        arch_interaction_op=None, # 初始的特征合并方式，dot cat
        arch_interaction_itself=False,
        sigmoid_bot=-1, #relu函数
        sigmoid_top=-1,
        sync_dense_params=True,             
        loss_threshold=0.0,
        ndevices=-1,
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold #0.0
            # create operators
            self.emb_l = self.create_emb(m_spa, ln_emb)
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

    def apply_mlp(self, x, layers):
        return layers(x)
    def apply_emb(self, lS_o, lS_i, emb_l):
        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]
            E = emb_l[k]
            V = E(sparse_index_group_batch, sparse_offset_group_batch)
            ly.append(V)
        return ly
    def interact_features(self, x, ly):
        if self.arch_interaction_op == "dot":
           # print("错误位置：{}:,x=={}+++ly=={}".format(95,x,ly))
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2)) #两个batch矩阵相乘
            _, ni, nj = Z.shape
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )
        return R
    def forward(self, dense_x, lS_o, lS_i):
        return self.sequential_forward(dense_x, lS_o, lS_i)
    def sequential_forward(self, dense_x, lS_o, lS_i):
        x = self.apply_mlp(dense_x, self.bot_l) #在这考虑构建的网络是否匹配输入的数据
        ly = self.apply_emb(lS_o, lS_i, self.emb_l)
        z = self.interact_features(x, ly)
        p = self.apply_mlp(z, self.top_l)
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:#loss_threshold=0.0
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p
        return z

arch_mlp_bot="4-512-256-128-30"
arch_mlp_top="1024-512-512-256-1"
ln_bot = np.fromstring(arch_mlp_bot, dtype=int, sep="-")
ln_emb = train_data.counts
m_den = train_data.m_den
ln_bot[0] = m_den
m_spa = 30 # 隐藏特征大小
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
nepochs=10
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
    AUC_=[]
    '''测试部分'''
    for (X, lS_o, lS_i,T) in test_loader:
        X=X.cuda(1)
        lS_o=[S_o.cuda(1) for S_o in lS_o]
        lS_i= [S_i.cuda(1) for S_i in lS_i]
        
        Z = dlrm(X, lS_o, lS_i)
        S = Z.detach().cpu().numpy()
        T = T.detach().cpu().numpy()
        auc=roc_auc_score(T,S)
        AUC_.append(auc)
    print("第{}轮测试集AUC : {}".format(k,sum(AUC_)/len(AUC_)))
    
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
