from transformers import AutoTokenizer, AutoModelForMaskedLM
import tensorflow as tf


class DrugEmbeddingModel(tf.keras.Model):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def call(self, inputs):
        attention_mask = inputs["attention_mask"]
        outputs = self.encoder(**inputs)
        last_hidden = outputs.last_hidden_state
        mask = tf.cast(attention_mask, tf.float32)
        mol_embeddings = tf.reduce_sum(last_hidden * tf.expand_dims(mask, -1), axis=1) / tf.reduce_sum(mask, axis=1, keepdims=True)
        return mol_embeddings


def get_drug_embedding_model():
    # Load model directly
    model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    drug_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    return model, drug_tokenizer
