from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

def get_target_embedding_model(seq_len=512):
    pretrained_model_generator, target_tokenizer = load_pretrained_model()
    target_encoder = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(seq_len))
    return target_encoder, target_tokenizer
