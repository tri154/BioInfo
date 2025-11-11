import tensorflow as tf
from drug_embedding import get_drug_embedding_model
from target_embedding import get_target_embedding_model
from tensorflow.keras.layers import Concatenate, Dense
from tensorflow.keras.models import Model
from dataloader import DrugMANDataset
import keras
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import numpy as np
import copy

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


def evaluate(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    f1 = (2 * precision * recall)/(precision+recall)
    best_f1 = np.max(f1[np.isfinite(f1)])
    return roc_auc, pr_auc, best_f1


def test(model, test_set, prepare_batch):
    y_pred, y_label = [], []
    for drugs, targets, label in test_set:
        inputs, label = prepare_batch(drugs, targets, label)
        probs = model(inputs, training=False)
        probs = tf.squeeze(probs, axis=-1)
        temp = probs.numpy().tolist()
        y_pred = y_pred + temp
        y_label = y_label + label.numpy().tolist()
        a = input("stop")
        if a == '1':
            break
    roc_auc, pr_auc, best_f1 = evaluate(y_label, y_pred)
    return roc_auc, pr_auc, best_f1

def main_old():
    epochs = 1
    lr = 3e-5
    batch_size = 2
    seq_len = 5179

    model, drug_tokenizer, target_tokenizer = get_embedding_model(seq_len)
    def prepare_batch(drugs, targets, label, seq_len=seq_len):
        drugs = [d.numpy().decode('utf-8') for d in drugs]
        targets = [t.numpy().decode('utf-8') for t in targets]

        drug_inputs = drug_tokenizer(drugs, padding=True, truncation=True, return_tensors="tf")
        target_inputs = target_tokenizer.encode_X(targets, seq_len)
        target_inputs = [tf.convert_to_tensor(x) for x in target_inputs]
        inputs = [drug_inputs['input_ids'], drug_inputs['attention_mask'], target_inputs[0], target_inputs[1]]
        return inputs, label

    dataFolder = 'data/warm_start'
    # dataFolder = 'data/cold_start'
    embFolder = 'data/bionic_embed'
    dataset = DrugMANDataset(dataFolder, embFolder)
    train_set, val_set, test_set = dataset.get_dataset()
    train_set = train_set.shuffle(2048).batch(batch_size, drop_remainder=True)
    val_set = val_set.batch(batch_size)
    test_set = test_set.batch(batch_size)

    loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = keras.optimizers.AdamW(learning_rate=lr)
    best_val_auroc = 0
    best_epoch = -1
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        for step, (drugs, targets, label) in enumerate(train_set):
            inputs, label = prepare_batch(drugs, targets, label)
            with tf.GradientTape() as tape:
                probs = model(inputs, training=True)
                loss_value = loss_fn(label, probs)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))
        val_auroc, val_auprc, _ = test(model, val_set, prepare_batch)
        print("Epoch " + str(epoch))
        # print("train_loss: %.8f" % (epoch_loss))
        print("val_AUROC: %.4f, val_AUPRC: %.4f" % (val_auroc, val_auprc,))
        if val_auroc > best_val_auroc:
            best_model = copy.deepcopy(model)
            best_val_auroc = val_auroc
            best_epoch = epoch

    print("Best_epoch  " + str(best_epoch))
    test_auroc, test_auprc, _ = test(best_model, test_set, prepare_batch)
    print("test_auroc: %.4f,test_auprc: %.4f ." % (test_auroc, test_auprc))

if __name__ == "__main__":
    main_old()
    # example_target()
