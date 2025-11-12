import tensorflow as tf
from drug_embedding import get_drug_embedding_model
from target_embedding import get_target_embedding_model
import torch
import pandas as pd

def example_drug(smiles):
    # smiles = ["Cc1ccccc1", "CCO", "C1=CC=CC=C1O"]
    drug_encoder, drug_tokenizer = get_drug_embedding_model()
    inputs = drug_tokenizer(smiles, padding=True, truncation=True)
    mask = torch.tensor(inputs['attention_mask'])
    outputs = drug_encoder(torch.tensor(inputs['input_ids']), torch.tensor(inputs['attention_mask']), output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]
    mol_embeddings = torch.sum(last_hidden * torch.unsqueeze(mask, dim=-1), dim=1) / torch.sum(mask, dim=1, keepdim=True)
    return mol_embeddings

def example_target():
    seqs = list()
    seqs.append("MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL")
    seqs.append("MDDADPEERNYDNMLKMLSDLNKDLEKLLEEMEKISVQATWMAYDMVVMRTNPTLAESMRRLEDAFVNCKEEMEKNWQELLHETKQRL")
    seq_len = 512
    target_encoder, target_tokenizer = get_target_embedding_model(seq_len)
    inputs = target_tokenizer.encode_X(seqs, seq_len)
    inputs = [tf.convert_to_tensor(x) for x in inputs]
    local_emb, global_emb = target_encoder(inputs, training=True)
    print(global_emb.shape)

from dataloader import DrugMANDataset
def main_drug():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataFolder = 'data/warm_start'
    # dataFolder = 'data/cold_start'
    embFolder = 'data/bionic_embed'
    dataset = DrugMANDataset(dataFolder, embFolder)
    all_binds, drugs, targets = dataset.get_all_binds()
    drug_encoder, drug_tokenizer = get_drug_embedding_model()
    drug_encoder.to(device)
    drug_encoder.eval()
    drug_embs = list()
    for d in drugs['rdkit_smile']:
        inputs = drug_tokenizer([d], padding=True, truncation=True)
        input_ids = torch.tensor(inputs['input_ids'], device=device)
        mask = torch.tensor(inputs['attention_mask'], device=device)
        with torch.no_grad():
            outputs = drug_encoder(input_ids, mask, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            mol_embeddings = torch.sum(last_hidden * torch.unsqueeze(mask, dim=-1), dim=1) / torch.sum(mask, dim=1, keepdim=True)
        drug_embs.append(mol_embeddings)
    drug_embs = torch.concat(drug_embs, dim=0)
    pub_chemcids = drugs['pubchem_cid']
    emb_df = pd.DataFrame(drug_embs.cpu().numpy(), index=pub_chemcids.values)
    emb_df.index.name = "pubchem_cid"
    emb_df.to_csv("drug_features.tsv", sep="\t", float_format="%.6f")

main_drug()
