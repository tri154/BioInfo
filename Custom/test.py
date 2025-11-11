from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import tensorflow as tf

seq = list()
seq.append("MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL")
seq.append("MDDADPEERNYDNMLKMLSDLNKDLEKLLEEMEKISVQATWMAYDMVVMRTNPTLAESMRRLEDAFVNCKEEMEKNWQELLHETKQRL")
seq_len = 298 + 2

pretrained_model_generator, input_encoder = load_pretrained_model()
model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(seq_len))
# model = pretrained_model_generator.create_model(seq_len)
with tf.GradientTape() as tape:
    inputs = input_encoder.encode_X(seq, seq_len)
    inputs = [tf.convert_to_tensor(x) for x in inputs]
    # local_emb, global_emb = model.predict(X, batch_size = len(seq))
    local_emb, global_emb = model(inputs, training=True)
    output = tf.reduce_mean(global_emb)

grads = tape.gradient(output, model.trainable_variables)
any_grad = any([g is not None for g in grads])
print("Gradients available:", any_grad)
