from dataset import parse_words, Dataset
import constants
from data_utils import make_triple_vocab, load_vocab, get_trimmed_w2v_vectors
import pickle
import tensorflow as tf
from collections import defaultdict

vocab_poses = load_vocab(constants.ALL_POSES)
vocab_synsets = load_vocab(constants.ALL_SYNSETS)
vocab_rels = load_vocab(constants.ALL_DEPENDS)
vocab_words = load_vocab(constants.ALL_WORDS)

# datasets = ['train', 'dev', 'test']
# all_rels = []

# for dataset in datasets:
#     with open(constants.RAW_DATA + 'sdp_data_acentors_bert.' + dataset + '.txt') as f:
#         lines = f.readlines()
#         words, poses, synsets, labels, identities, triples, positions = parse_words(lines)
#         print(max([len(r) for r in words]))
        # for rel in relations:
        #     all_rels.extend(rel)
#
# all_rels = list(set(all_rels))
# with open(constants.ALL_DEPENDS, 'w') as f1:
#     for rel in all_rels:
#         f1.write(rel)
#         f1.write('\n')
# print(relations)

# print(words)

chem_vocab = make_triple_vocab(constants.DATA + 'chemical2id.txt')
dis_vocab = make_triple_vocab(constants.DATA + 'disease2id.txt')

train = Dataset(constants.RAW_DATA + 'sentence_data_acentors.train.txt',
                constants.RAW_DATA + 'sdp_data_acentors_bert.train.txt',
                vocab_words=vocab_words,
                vocab_poses=vocab_poses,
                vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab, vocab_dis=dis_vocab)
pickle.dump(train, open(constants.PICKLE_DATA + 'train.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

dev = Dataset(constants.RAW_DATA + 'sentence_data_acentors.dev.txt',
              constants.RAW_DATA + 'sdp_data_acentors_bert.dev.txt',
              vocab_words=vocab_words,
              vocab_poses=vocab_poses,
              vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab, vocab_dis=dis_vocab)
pickle.dump(dev, open(constants.PICKLE_DATA + 'dev.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

test = Dataset(constants.RAW_DATA + 'sentence_data_acentors.test.txt',
               constants.RAW_DATA + 'sdp_data_acentors_bert.test.txt',
               vocab_words=vocab_words,
               vocab_poses=vocab_poses,
               vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab, vocab_dis=dis_vocab)
pickle.dump(test, open(constants.PICKLE_DATA + 'test.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

# Train, Validation Split
validation = Dataset('', '', process_data=False)
train_ratio = 0.85
n_sample = int(len(dev.words) * (2 * train_ratio - 1))
props = ['words', 'head_mask', 'e1_mask', 'e2_mask', 'relations', 'labels', 'poses', 'synsets', 'identities',
         'triples']

for prop in props:
    train.__dict__[prop].extend(dev.__dict__[prop][:n_sample])
    validation.__dict__[prop] = dev.__dict__[prop][n_sample:]

train.get_padded_data()
validation.get_padded_data()

# print(train.relations)
print(train.words)

# wn_emb = get_trimmed_w2v_vectors('data/w2v_model/wordnet_embeddings.npz')
#
# # print(wn_emb.shape)
#
# with open(constants.EMBEDDING_CHEM, 'rb') as f:
#     chem_emb = pickle.load(f)
#     f.close()
#
# with open(constants.EMBEDDING_DIS, 'rb') as f:
#     dis_emb = pickle.load(f)
#     f.close()
#
# concated = tf.concat([chem_emb, dis_emb], axis=0)
# print(concated)
