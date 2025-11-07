import numpy as np
import torch
from toolkit.dataloader import DrugMANDataset
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def aac(data):
    aa = 'ACDEFGHIKLMNPQRSTVWY'
    res = list()
    for sequence in data:
        count = Counter(sequence)
        total = len(sequence)
        res.append([count.get(a, 0) / total for a in aa])
    return np.array(res)

def ecfp4_new(data, nbits=1024):
    generator = GetMorganGenerator(radius=2, fpSize=nbits)
    res = list()
    for smile in data:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            fp = np.zeros(nbits, dtype=int)
        else:
            fp = generator.GetFingerprint(mol)
        res.append(fp)
    return np.array(res)

def ecfp4(data, nbits=1024):
    res = list()
    for smile in data:
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits)
        res.append(fp)
    return np.array(res)

def id2seq(df, all_binds):
    drug_useq = all_binds[["pubchem_cid", "rdkit_smile"]]
    drug_useq = drug_useq.drop_duplicates()

    target_useq = all_binds[["gene_id", "sequence"]]
    target_useq = target_useq.drop_duplicates()

    drug_seqs = pd.merge(df["pubchem_cid"], drug_useq, on=["pubchem_cid"])
    target_seqs = pd.merge(df["gene_id"], target_useq, on=["gene_id"])
    return drug_seqs["rdkit_smile"].to_numpy(), target_seqs["sequence"].to_numpy()


# main
def main():
    nbits = 512

    # dataFolder = 'data/warm_start'
    dataFolder = 'data/cold_start'
    dataset = DrugMANDataset(dataFolder, None)
    df_train, df_val, df_test = dataset.load_data()
    all_binds = pd.read_csv("data/all_bind.csv")

    train_drug, train_target = id2seq(df_train, all_binds)
    test_drug, test_target = id2seq(df_test, all_binds)

    train_drug_emb = ecfp4_new(train_drug, nbits)
    train_target_emb= aac(train_target)
    x_train = np.concatenate([train_drug_emb, train_target_emb], axis=1)

    test_drug_emb = ecfp4_new(test_drug, nbits)
    test_target_emb = aac(test_target)
    x_test = np.concatenate([test_drug_emb, test_target_emb], axis=1)

    y_train = df_train["label"]
    y_test = df_test["label"]

    # rf = RandomForestClassifier(
    #     n_estimators=300,
    #     max_depth=None,
    #     random_state=42,
    #     n_jobs=-1
    # ) # 840, 867

    # rf = XGBClassifier(
    #     n_estimators=300,
    #     learning_rate=0.05,
    #     max_depth=6,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     random_state=42,
    #     n_jobs=-1,
    #     eval_metric='logloss'
    # ) # 845, 870

    # rf = LGBMClassifier(
    #     n_estimators=300,
    #     learning_rate=0.05,
    #     max_depth=-1,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     random_state=42,
    #     n_jobs=-1
    # ) # 841, 867

    rf = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        verbose=0
    )

    rf.fit(x_train, y_train)

    y_pred_prob = rf.predict_proba(x_test)[:, 1]

    auroc = roc_auc_score(y_test, y_pred_prob)
    auprc = average_precision_score(y_test, y_pred_prob)

    print("auroc: ", auroc, "auprc: ", auprc)

main()
