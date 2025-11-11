import tensorflow as tf
from drug_embedding import get_drug_embedding_model
from target_embedding import get_target_embedding_model
from tensorflow.keras.layers import Concatenate, Dense
from tensorflow.keras.models import Model
from dataloader import DrugMANDataset

def example_drug():
    drug_encoder, drug_tokenizer = get_drug_embedding_model()
    smiles = ["Cc1ccccc1", "CCO", "C1=CC=CC=C1O"]
    inputs = drug_tokenizer(smiles, padding=True, truncation=True, return_tensors="tf")
    inputs = [inputs["input_ids"], inputs["attention_mask"]]
    outputs = drug_encoder(inputs, training=True)
    print(outputs.shape)

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


def example_both():
    smiles = ["Cc1ccccc1", "CCO"]
    seqs = list()
    seqs.append("MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL")
    seqs.append("MDDADPEERNYDNMLKMLSDLNKDLEKLLEEMEKISVQATWMAYDMVVMRTNPTLAESMRRLEDAFVNCKEEMEKNWQELLHETKQRL")

    seq_len = 512
    model, drug_tokenizer, target_tokenizer = get_embedding_model(seq_len) # drug_ids, attention_mask, target_seq, target_annotations

    drug_inputs = drug_tokenizer(smiles, padding=True, truncation=True, return_tensors="tf")
    target_inputs = target_tokenizer.encode_X(seqs, seq_len)
    target_inputs = [tf.convert_to_tensor(x) for x in target_inputs]

    inputs = [drug_inputs['input_ids'], drug_inputs['attention_mask'], target_inputs[0], target_inputs[1]]
    outs = model(inputs, training=False)
    print(outs.shape)

def get_embedding_model(seq_len=512):
    drug_encoder, drug_tokenizer = get_drug_embedding_model()
    target_encoder, target_tokenizer = get_target_embedding_model(seq_len)

    drug_embs = drug_encoder.outputs[0]
    _, target_embs = target_encoder.outputs

    combined = Concatenate()([drug_embs, target_embs])
    output = Dense(1, activation='sigmoid')(combined)
    model = Model(inputs=drug_encoder.inputs + target_encoder.inputs, outputs=output)
    return model, drug_tokenizer, target_tokenizer


if __name__ == "__main__":
    model, drug_tokenizer, target_tokenizer = get_embedding_model()

    def batch_preprocess(drug, target, label, seq_len=512):
        print(drug)
        print(target)
        input()
        drug_inputs = drug_tokenizer(drug, padding=True, truncation=True, return_tensors="tf")

        target_inputs = target_tokenizer.encode_X(target, seq_len)
        target_inputs = [tf.convert_to_tensor(x) for x in target_inputs]
        inputs = [drug_inputs['input_ids'], drug_inputs['attention_mask'], target_inputs[0], target_inputs[1]]
        return inputs

    dataFolder = 'data/warm_start'
    # dataFolder = 'data/cold_start'
    embFolder = 'data/bionic_embed'
    dataset = DrugMANDataset(dataFolder, embFolder)
    train_set, val_set, test_set = dataset.get_dataset()
    train_set = train_set.map(batch_preprocess).shuffle(100).batch(50, drop_remainder=True)
    val_set = val_set.map(batch_preprocess).batch(50)
    test_set = test_set.map(batch_preprocess).batch(50)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        train_set,
        validation_data=val_set,
        epochs=20
    )
