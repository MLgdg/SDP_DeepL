import torch 
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from numpy import random as ra
from os import  path
from sklearn.metrics import  roc_auc_score 


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