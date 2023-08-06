import sentencepiece as sp


# Params
n_tokens = 1000

# Train sentencepiece model on input data
sp.SentencePieceTrainer.train(f'--input=data/cankar/cankar-proza.txt \
                                --model_prefix=cankar-tokens \
                                --vocab_size={n_tokens}')

# Make segmenter instance and load the model
token_model = sp.SentencePieceProcessor()
token_model.load('cankar-tokens.model')

# Load Cankar text
with open('data/cankar/cankar-proza.txt') as f:
    text = f.read()

# Check tokenization
print('Encoded as pieces:')
print(token_model.encode_as_pieces(text[:100]))

print('Encoded as IDs:')
print(token_model.encode_as_ids(text[:100]))

# Test reencoding
ids = token_model.decode_ids(token_model.encode_as_ids(text[:100]))
