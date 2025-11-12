import pandas as pd
import numpy as np
import torch.utils.data as Data
import torch
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


class DrugMANDataset:
    def __init__(self, file_csv, file_emb):
        self.file_csv = file_csv
        self.file_emb = file_emb
        self.all_binds = "data/all_bind.csv"

    def load_data(self):
        part = 1
        train = pd.read_csv(self.file_csv + f"/bind_train_part{part}.csv")
        val = pd.read_csv(self.file_csv + f"/bind_val_part{part}.csv")
        test = pd.read_csv(self.file_csv + f"/bind_test_part{part}.csv")

        return train, val, test

    def load_embed(self):
        drug_emb = pd.read_csv(self.file_emb + "/drug_features.tsv", index_col=0, delimiter="\t")
        target_emb = pd.read_csv(self.file_emb + "/target_features.tsv", index_col=0, delimiter="\t")

        return drug_emb, target_emb

    def get_all_binds(self, ):
        all_binds = pd.read_csv(self.all_binds)
        all_binds_drug = all_binds[['pubchem_cid', 'rdkit_smile']].drop_duplicates(subset='pubchem_cid')
        all_binds_target = all_binds[['gene_id', 'sequence']].drop_duplicates(subset='gene_id')
        return all_binds, all_binds_drug, all_binds_target

    def get_dataset(self):
        all_binds = pd.read_csv(self.all_binds)
        all_binds_drug = all_binds[['pubchem_cid', 'rdkit_smile']].drop_duplicates(subset='pubchem_cid')
        all_binds_target = all_binds[['gene_id', 'sequence']].drop_duplicates(subset='gene_id')

        train, val, test = self.load_data()
        # train = train.sample(frac=1, random_state=42) # shuffle

        train_drug_seqs = train.merge(all_binds_drug[['pubchem_cid', 'rdkit_smile']], on='pubchem_cid', how='left')['rdkit_smile']
        train_target_seqs = train.merge(all_binds_target[['gene_id', 'sequence']], on='gene_id', how='left')['sequence']

        val_drug_seqs = val.merge(all_binds_drug[['pubchem_cid', 'rdkit_smile']], on='pubchem_cid', how='left')['rdkit_smile']
        val_target_seqs = val.merge(all_binds_target[['gene_id', 'sequence']], on='gene_id', how='left')['sequence']

        test_drug_seqs = test.merge(all_binds_drug[['pubchem_cid', 'rdkit_smile']], on='pubchem_cid', how='left')['rdkit_smile']
        test_target_seqs = test.merge(all_binds_target[['gene_id', 'sequence']], on='gene_id', how='left')['sequence']

        train_drug_seqs = np.array(train_drug_seqs)
        train_target_seqs = np.array(train_target_seqs)

        val_drug_seqs = np.array(val_drug_seqs.values)
        val_target_seqs = np.array(val_target_seqs)

        test_drug_seqs = np.array(test_drug_seqs)
        test_target_seqs = np.array(test_target_seqs)

        train_label = np.array(train['label'])
        val_label = np.array(val['label'])
        test_label = np.array(test['label'])

        train_set = (train_drug_seqs, train_target_seqs, train_label)
        val_set = (val_drug_seqs, val_target_seqs, val_label)
        test_set = (test_drug_seqs, test_target_seqs, test_label)

        return train_set, val_set, test_set


    def get_dataloader(self):
        train, val, test = self.load_data()
        drug_emb, target_emb = self.load_embed()

        train_drug_emb = drug_emb.loc[train['pubchem_cid'], ]
        train_target_emb = target_emb.loc[train['gene_id'], ]

        val_drug_emb = drug_emb.loc[val["pubchem_cid"], ]
        val_target_emb = target_emb.loc[val['gene_id'], ]

        test_drug_emb = drug_emb.loc[test["pubchem_cid"], ]
        test_target_emb = target_emb.loc[test['gene_id'], ]

        # normalized by z-score and convert to tensor type
        scaler = StandardScaler()
        train_drug_emb = scaler.fit_transform(np.array(train_drug_emb))
        train_target_emb = scaler.fit_transform(np.array(train_target_emb))
        train_drug_emb = torch.FloatTensor(train_drug_emb)
        train_target_emb = torch.FloatTensor(train_target_emb)
        train_label = torch.FloatTensor(np.array(train['label']))

        val_drug_emb = scaler.fit_transform(np.array(val_drug_emb))
        val_target_emb = scaler.fit_transform(np.array(val_target_emb))
        val_drug_emb = torch.FloatTensor(val_drug_emb)
        val_target_emb = torch.FloatTensor(val_target_emb)
        val_label = torch.FloatTensor(np.array(val['label']))

        test_drug_emb = scaler.fit_transform(np.array(test_drug_emb))
        test_target_emb = scaler.fit_transform(np.array(test_target_emb))
        test_drug_emb = torch.FloatTensor(test_drug_emb)
        test_target_emb = torch.FloatTensor(test_target_emb)
        test_label = torch.FloatTensor(np.array(test['label']))

        # create dataloader
        train_dataset = Data.TensorDataset(train_drug_emb, train_target_emb, train_label)
        val_dataset = Data.TensorDataset(val_drug_emb, val_target_emb, val_label)
        test_dataset = Data.TensorDataset(test_drug_emb, test_target_emb, test_label)

        params = {'batch_size': 512, 'shuffle': True, 'num_workers': 0, 'drop_last': True}
        if train_dataset or val_dataset:
            train_loader = Data.DataLoader(train_dataset, **params)
            val_loader = Data.DataLoader(val_dataset, **params)
        if test_dataset:
            params['shuffle'] = False
            params['drop_last'] = False
            params['batch_size'] = len(test_dataset)
            test_loader = Data.DataLoader(test_dataset, **params)
            test_bcs = len(test_dataset)

        return train_loader, val_loader, test_loader, test_bcs
