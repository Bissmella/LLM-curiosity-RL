from enum import Enum

from transformers import AutoProcessor, AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig,Idefics2ForConditionalGeneration,AutoModelForCausalLM
 
class ModelTypesEnum(Enum):
    causal = AutoModelForCausalLM
    seq2seq = AutoModelForSeq2SeqLM


def load_hf_vlm_and_tokenizer(type, path, pretrained):
    print("Loading vlm model {}".format(path))
    #tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

    # Select class according to type
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    
    
    model_class = ModelTypesEnum[type].value
    if pretrained:
        model_method = lambda **kwargs: model_class.from_pretrained(path, **kwargs)
    else:
        model_method = lambda **kwargs: model_class.from_config(config, **kwargs)

    return processor, model_method
