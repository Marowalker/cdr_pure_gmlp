from dataset import Dataset
import constants
from data_utils import make_triple_vocab, load_vocab, get_trimmed_w2v_vectors
import pickle
import tensorflow as tf
from evaluate.bc5 import evaluate_bc5
from gmlp.model.cdr_gmlp import CDRgMLPModel


def main():
    if constants.IS_REBUILD == 1:
        print('Build data')
        vocab_words = load_vocab(constants.ALL_WORDS)
        vocab_poses = load_vocab(constants.ALL_POSES)
        vocab_synsets = load_vocab(constants.ALL_SYNSETS)
        vocab_rels = load_vocab(constants.ALL_DEPENDS)
        chem_vocab = make_triple_vocab(constants.DATA + 'chemical2id.txt')
        dis_vocab = make_triple_vocab(constants.DATA + 'disease2id.txt')

        train = Dataset(constants.RAW_DATA + 'sdp_data_acentors_bert.train.txt',
                        vocab_words=vocab_words,
                        vocab_poses=vocab_poses,
                        vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab, vocab_dis=dis_vocab)
        pickle.dump(train, open(constants.PICKLE_DATA + 'train.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

        dev = Dataset(constants.RAW_DATA + 'sdp_data_acentors_bert.dev.txt',
                      vocab_words=vocab_words,
                      vocab_poses=vocab_poses,
                      vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab, vocab_dis=dis_vocab)
        pickle.dump(dev, open(constants.PICKLE_DATA + 'dev.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

        test = Dataset(constants.RAW_DATA + 'sdp_data_acentors_bert.test.txt',
                       vocab_words=vocab_words,
                       vocab_poses=vocab_poses,
                       vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab, vocab_dis=dis_vocab)
        pickle.dump(test, open(constants.PICKLE_DATA + 'test.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

    else:
        print('Load data')
        train = pickle.load(open(constants.PICKLE_DATA + 'train.pickle', 'rb'))
        dev = pickle.load(open(constants.PICKLE_DATA + 'dev.pickle', 'rb'))
        test = pickle.load(open(constants.PICKLE_DATA + 'test.pickle', 'rb'))

    # Train, Validation Split
    validation = Dataset('', '', process_data=False)
    train_ratio = 0.85
    n_sample = int(len(dev.words) * (2 * train_ratio - 1))
    props = ['words', 'labels', 'poses', 'synsets', 'identities', 'triples']

    for prop in props:
        train.__dict__[prop].extend(dev.__dict__[prop][:n_sample])
        validation.__dict__[prop] = dev.__dict__[prop][n_sample:]

    train.get_padded_data()
    validation.get_padded_data()
    test.get_padded_data(shuffled=False)

    print("Train shape: ", len(train.words))
    print("Test shape: ", len(test.words))
    print("Validation shape: ", len(validation.words))

    wn_emb = get_trimmed_w2v_vectors('data/w2v_model/wordnet_embeddings.npz')

    with open(constants.EMBEDDING_CHEM, 'rb') as f:
        chem_emb = pickle.load(f)
        f.close()

    with open(constants.EMBEDDING_DIS, 'rb') as f:
        dis_emb = pickle.load(f)
        f.close()

    with open('data/w2v_model/transe_cdr_word_16.pkl', 'rb') as f:
        word_emb = pickle.load(f)
        f.close()

    with open('data/w2v_model/transe_cdr_relation_16.pkl', 'rb') as f:
        rel_emb = pickle.load(f)
        f.close()

    model = CDRgMLPModel(depth=36, chem_emb=chem_emb, dis_emb=dis_emb,
                         wordnet_emb=wn_emb, cdr_emb=word_emb, rel_emb=rel_emb)
    model.build(train, validation)

    y_pred = model.predict(test)
    answer = {}
    identities = test.identities

    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            if identities[i][0] not in answer:
                answer[identities[i][0]] = []

            if identities[i][1] not in answer[identities[i][0]]:
                answer[identities[i][0]].append(identities[i][1])

    print(
        'result: abstract: ', evaluate_bc5(answer)
    )


if __name__ == '__main__':
    main()
