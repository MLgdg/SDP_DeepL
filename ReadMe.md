## 输入数据结构以及模型结构
数据由类别数据和数值型数据构成，其次是标签数据，  
例如：   
traindata：  
intdata   catdata    target  
2          A         0    
3          B         1  
4          C         0  
7          D         1  
              
### 模型结构

SDP_L(   
  (emb): ModuleList(    
    (0): EmbeddingBag(6, 3, mode=sum)     
    (1): EmbeddingBag(7, 3, mode=sum)  
    (2): EmbeddingBag(6, 3, mode=sum)  
    (3): EmbeddingBag(5, 3, mode=sum)  
    (4): EmbeddingBag(8, 3, mode=sum)  
    (5): EmbeddingBag(6, 3, mode=sum)  
    (6): EmbeddingBag(5, 3, mode=sum)  
  )  
  (ann): Sequential(  
    (0): Linear(in_features=5, out_features=2, bias=True)  
    (1): ReLU()   
    (2): Linear(in_features=2, out_features=3, bias=True)    
    (3): ReLU()  
  )  
  (ann): Sequential(  
    (0): Linear(in_features=31, out_features=4, bias=True)  
    (1): ReLU()  
    (2): Linear(in_features=4, out_features=3, bias=True)  
    (3): ReLU()  
    (4): Linear(in_features=3, out_features=1, bias=True)  
    (5): Sigmoid()  
  )  
)  
