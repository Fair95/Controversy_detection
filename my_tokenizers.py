from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizerFast, PreTrainedTokenizerFast


def construct_tokenizer(texts, batch_size):

    def batch_iterator():
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]

    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.pre_tokenizer.pre_tokenize_str("This is an example!")
    special_tokens = ["[UNK]", "[PAD]"]
    trainer = trainers.WordLevelTrainer(vocab_size=25000, special_tokens=special_tokens)
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    new_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]",
        sep_token="[SEP]", cls_token="[CLS]", mask_token="[MASK]" )

    return new_tokenizer

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('cleaned_2.csv')
    tk = construct_tokenizer(df.paragraph, 100)
    print([tk(sent, truncation=True, padding='max_length', max_length=512, return_tensors='pt') for sent in df.paragraph])

