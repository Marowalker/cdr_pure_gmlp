import numpy as np
from nltk.corpus import wordnet as wn
import constants
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
from tensorflow.keras.preprocessing.text import Tokenizer

np.random.seed(13)


def clean_lines(lines):
    cleaned_lines = []
    for line in lines:
        l = line.strip().split()
        if len(l) == 1:
            cleaned_lines.append(line)
        else:
            pair = l[0]
            if '-1' not in pair:
                cleaned_lines.append(line)
    return cleaned_lines


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def my_pad_sequences(sequences, pad_tok, max_sent_length, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        max_sent_length: the maximum length of the padded sentence
        dtype: the type of the final return value
        nlevels: the level (no. of dimensions) of the padded matrix
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        # max_length = max(map(lambda x: len(x), sequences))
        max_length = max_sent_length
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))

        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_sent_length)

    return np.array(sequence_padded), sequence_length


def parse_words(raw_data):
    raw_data = clean_lines(raw_data)
    all_words = []
    all_poses = []
    all_synsets = []
    # all_relations = []
    all_positions = []
    all_labels = []
    all_identities = []
    all_triples = []
    pmid = ''
    for line in raw_data:
        l = line.strip().split()
        if len(l) == 1:
            pmid = l[0]
        else:
            pair = l[0]
            label = l[1]
            if label:
                chem, dis = pair.split('_')
                all_triples.append([chem, dis])

                joint_sdp = ' '.join(l[2:])
                sdps = joint_sdp.split("-PUNC-")
                for sdp in sdps:
                    # S xuoi
                    nodes = sdp.split()
                    words = []
                    poses = []
                    synsets = []
                    positions = []
                    # relations = []
                    for idx, node in enumerate(nodes):
                        node = node.split('|')
                        if idx % 2 == 0:
                            for n_idx, _node in enumerate(node):
                                word = constants.UNK if _node == '' else _node
                                if n_idx == 0:
                                    w, p, s = word.split('\\')
                                    p = 'NN' if p == '' else p
                                    s = str(wn.synsets('entity')[0].offset()) if s == '' else s
                                    _w, position = w.rsplit('_', 1)
                                    # _w = w
                                    words.append(_w)
                                    poses.append(p)
                                    synsets.append(s)
                                    positions.append(min(int(position), constants.MAX_LENGTH))
                                else:
                                    w = word.split('\\')[0]
                        else:
                            rel = '(' + node[0].strip().split('_')[-1]
                            # print(node)
                            words.append(rel)
                            # relations.append(rel)

                    all_words.append(words)
                    all_poses.append(poses)
                    all_synsets.append(synsets)
                    all_positions.append(positions)
                    # all_relations.append(relations)
                    all_labels.append([label])
                    all_identities.append((pmid, pair))
            else:
                print(l)

    return all_words, all_poses, all_synsets, all_labels, all_identities, all_triples, all_positions


class Dataset:
    def __init__(self, sdp_name, vocab_words=None, vocab_poses=None, vocab_synset=None, vocab_rels=None,
                 vocab_chems=None,
                 vocab_dis=None,
                 process_data=True):
        self.sdp_name = sdp_name

        self.vocab_words = vocab_words
        self.vocab_poses = vocab_poses
        self.vocab_synsets = vocab_synset
        self.vocab_rels = vocab_rels

        self.vocab_chems = vocab_chems
        self.vocab_dis = vocab_dis

        if process_data:
            self._process_data()
            self._clean_data()

    def get_padded_data(self, shuffled=True):
        self._pad_data(shuffled=shuffled)

    def _clean_data(self):
        del self.vocab_poses
        del self.vocab_synsets
        del self.vocab_rels

    def _process_data(self):
        with open(self.sdp_name, 'r') as f1:
            raw_sdp = f1.readlines()
        data_word_relations, data_pos, data_synsets, data_y, self.identities, data_triples, data_positions = \
            parse_words(raw_sdp)

        # positions_1 = []
        # positions_2 = []
        labels = []
        poses = []
        synsets = []
        relations = []
        all_ents = []

        # for i in range(len(data_positions)):
        #     position_1, position_2 = [], []
        #     e1 = data_positions[i][0]
        #     e2 = data_positions[i][-1]
        #     for po in data_positions[i]:
        #         position_1.append((po - e1 + constants.MAX_LENGTH) // 5 + 1)
        #         position_2.append((po - e2 + constants.MAX_LENGTH) // 5 + 1)
        #     positions_1.append(position_1)
        #     positions_2.append(position_2)

        for i in range(len(data_pos)):

            ps, ss = [], []

            for p, s in zip(data_pos[i], data_synsets[i]):
                if p in self.vocab_poses:
                    p_id = self.vocab_poses[p]
                else:
                    p_id = self.vocab_poses['NN']
                ps += [p_id]
                if s in self.vocab_synsets:
                    synset_id = self.vocab_synsets[s]
                else:
                    synset_id = self.vocab_synsets[constants.UNK]
                ss += [synset_id]

            poses.append(ps)
            synsets.append(ss)

            lb = constants.ALL_LABELS.index(data_y[i][0])
            labels.append(lb)

        for i in range(len(data_word_relations)):
            rs = []
            for r in data_word_relations[i]:
                if data_word_relations[i].index(r) % 2 == 0:
                    if r in self.vocab_words:
                        r_id = self.vocab_words[r]
                    else:
                        r_id = self.vocab_words[constants.UNK]
                else:
                    if r in self.vocab_rels:
                        r_id = len(self.vocab_words) + self.vocab_rels[r] + 1
                    else:
                        r_id = len(self.vocab_words) + self.vocab_rels[constants.UNK] + 1
                rs.append(r_id)
            relations.append(rs)

        self.words = relations
        self.labels = labels
        self.poses = poses
        self.synsets = synsets
        # self.positions_1 = positions_1
        # self.positions_2 = positions_2
        self.triples = self.parse_triple(data_triples)

    def parse_triple(self, all_triples):
        data_triples = []
        for c, d in all_triples:
            c_id = int(self.vocab_chems[c])
            d_id = int(self.vocab_dis[d]) + int(len(self.vocab_chems))
            data_triples.append([c_id, d_id])

        return data_triples

    def _pad_data(self, shuffled=True):
        if shuffled:
            word_shuffled, pos_shuffled, synset_shuffled, label_shuffled, triple_shuffled = shuffle(self.words,
                                                                                                    self.poses,
                                                                                                    self.synsets,
                                                                                                    self.labels,
                                                                                                    self.triples)
        else:
            word_shuffled, pos_shuffled, synset_shuffled, label_shuffled, triple_shuffled =\
                self.words, self.poses, self.synsets, self.labels, self.triples

        self.words = tf.constant(pad_sequences(word_shuffled, maxlen=constants.MAX_LENGTH, padding='post'))
        self.poses = tf.constant(pad_sequences(pos_shuffled, maxlen=constants.MAX_LENGTH, padding='post'))
        self.synsets = tf.constant(pad_sequences(synset_shuffled, maxlen=constants.MAX_LENGTH, padding='post'))
        self.labels = tf.keras.utils.to_categorical(label_shuffled)
        # self.positions_1 = tf.constant(pad_sequences(positions_1_shuffle, maxlen=constants.MAX_LENGTH, padding='post'))
        # self.positions_2 = tf.constant(pad_sequences(positions_2_shuffle, maxlen=constants.MAX_LENGTH, padding='post'))
        self.triples = tf.constant(triple_shuffled, dtype='int32')

