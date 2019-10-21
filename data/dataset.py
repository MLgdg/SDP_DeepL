import torch 
import torch.nn as nn
from torch.utils.data import Dataset
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
