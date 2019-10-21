import torch

def collate_wrapper(list_of_tuples):
    transposed_data = list(zip(*list_of_tuples))
    X_int = torch.stack(transposed_data[0], 0)
    X_cat = torch.stack(transposed_data[1], 0)
    T       = torch.stack(transposed_data[2], 0).view(-1,1)
    sz0 = X_cat.shape[0]
    sz1 = X_cat.shape[1]
    #if 0:
    #    lS_i = [X_cat[:, i].pin_memory() for i in range(sz1)]
    #    lS_o = [torch.tensor(range(sz0)).pin_memory() for _ in range(sz1)]
    #    return X_int.pin_memory(), lS_o, lS_i, T.pin_memory()
    #else:
    lS_i = [X_cat[:, i] for i in range(sz1)]
    lS_o = [torch.tensor(range(sz0)) for _ in range(sz1)]
    return X_int, lS_o, lS_i, T