'''
Import Huggingface bookcorpus Dataset and save as txt file
which can be used by the train.py script.
'''


from datasets import load_dataset

data = load_dataset('bookcorpus')
rows_load = 1_000_000
txt = '\n'.join(data['train'][:rows_load]['text'])
with open('data/bookcorpus.txt', 'w') as f:
    f.write(txt)
