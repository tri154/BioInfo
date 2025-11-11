# Load model directly
from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf


smiles = ["Cc1ccccc1", "CCO", "C1=CC=CC=C1O"]

tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
tf_model = TFAutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", from_pt=True)

inputs = tokenizer(smiles, padding=True, truncation=True, return_tensors="tf")
outputs = tf_model(**inputs)
last_hidden = outputs.last_hidden_state
print(last_hidden.shape)

mask = tf.cast(inputs["attention_mask"], tf.float32)
mol_embeddings = tf.reduce_sum(last_hidden * tf.expand_dims(mask, -1), axis=1) / tf.reduce_sum(mask, axis=1, keepdims=True)
print("Molecule embeddings shape:", mol_embeddings.shape)
