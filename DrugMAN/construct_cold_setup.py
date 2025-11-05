import pandas as pd
import numpy as np
import networkx as nx

from sklearn.model_selection import train_test_split


# if __name__ == "__main__":
if __name__ == "off":
    train, dev, test = list(), list(), list()
    df = pd.read_csv("data/all_bind.csv")
    total = df.shape[0]
    print("total: ", total)
    print()
    test_size = total * 0.2
    dev_size = total * 0.1
    train_size = total - test_size - dev_size

    chem = df["pubchem_cid"].to_numpy()
    gene = df["gene_id"].to_numpy()

    pair = np.stack([chem, gene], axis=1)

    def try_one(nsize):
        is_free = np.zeros_like(chem) == 0
        indices = np.arange(0, pair.shape[0])
        test_indices = np.random.choice(indices, size=nsize, replace=False)
        test = pair[test_indices]
        is_free[test_indices] = False
        test_chem = np.unique(test[:, 0])
        test_gene = np.unique(test[:, 1])

        for c in test_chem:
            is_free[c == pair[:, 0]] = False
        for g in test_gene:
            is_free[g == pair[:, 1]] = False
        remain = pair[is_free]
        return test, remain

    for i in range(1000):
        nsize = 1100
        test, remain = try_one(nsize)
        ratio = remain.shape[0] / nsize
        print(remain.shape)
        print(ratio)
        if ratio > 4:
            np.savetxt("test_cold.txt", test)
            np.savetxt("train_dev_cold.txt", remain)
            input("found")

def sample_negative(data):
    total = data.shape[0]
    uchem = np.unique(data[:, 0])
    udrug = np.unique(data[:, 1])
    negative = list()
    for i in range(total):
        while True:
            c = np.random.choice(uchem)
            d = np.random.choice(udrug)
            cd = np.array([c, d])
            in_data = (data == cd).all(axis=1).any()
            in_negative = (not np.array(negative).shape[0] == 0) and (np.array(negative) == cd).all(axis=1).any()
            if not in_data and not in_negative:
                negative.append(cd)
                break
            else:
                print("same")
    negative = np.array(negative)
    return negative


if __name__ == "__main__":
    df = pd.read_csv("data/all_bind.csv")
    chem = df["pubchem_cid"].to_numpy()
    gene = df["gene_id"].to_numpy()

    pair = np.stack([chem, gene], axis=1)

    test = np.loadtxt("test_cold.txt")
    train_dev_cold = np.loadtxt("train_dev_cold.txt")

    test_chem = np.unique(test[:, 0])
    test_gene = np.unique(test[:, 1])

    for c in test_chem:
        check = (c == train_dev_cold[:, 0]).any()
        if check:
            input("error")
    for g in test_gene :
        check = (g == train_dev_cold[:, 1]).any()
        if check:
            input("error")

    print('done, no error')

    train, dev = train_test_split(train_dev_cold, test_size=0.14, shuffle=True)

    train, dev, test
    train_neg = sample_negative(train)
    dev_neg = sample_negative(dev)
    test_neg = sample_negative(test)

    train_com = np.concatenate([train, train_neg], axis=0)
    dev_com = np.concatenate([dev, dev_neg], axis=0)
    test_com = np.concatenate([test, test_neg])

    train_labels = np.concatenate((np.ones_like(train[:, 0]), np.zeros_like(train_neg[:, 0])), axis=0).reshape(-1, 1)
    dev_labels = np.concatenate((np.ones_like(dev[:, 0]), np.zeros_like(dev_neg[:, 0])), axis=0).reshape(-1, 1)
    test_labels = np.concatenate((np.ones_like(test[:, 0]), np.zeros_like(test_neg[:, 0])), axis=0).reshape(-1, 1)

    train_save = pd.DataFrame(np.concatenate([train_com, train_labels], axis=1).astype(int), columns=["pubchem_cid", "gene_id", "label"])
    dev_save = pd.DataFrame(np.concatenate([dev_com, dev_labels], axis=1).astype(int), columns=["pubchem_cid", "gene_id", "label"])
    test_save = pd.DataFrame(np.concatenate([test_com, test_labels], axis=1).astype(int), columns=["pubchem_cid", "gene_id", "label"])

    train_save.to_csv("bind_train.csv", index=False)
    dev_save.to_csv("bind_val.csv", index=False)
    test_save.to_csv("bind_test.csv", index=False)
