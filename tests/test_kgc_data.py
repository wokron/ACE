from pathlib import Path

from flair.custom_data_loader import ColumnDataLoader
from flair.data import Sentence
from flair.datasets import KGCDataset, FB15K_237


def test_kgc_dataset_load():

    dataset = KGCDataset(
        Path("resources/dataset/fb15k_237/train_test.txt"),
    )
    assert len(dataset) == 100

    item0 = dataset[0]
    assert type(item0) == Sentence

    head_id, relation, tail_id = item0.head_id, item0, item0.tail_id
    assert type(head_id) == int and type(relation) == Sentence and type(tail_id) == int
    print(relation)

    item99 = dataset[99]
    assert type(item99) == Sentence

    head_id, relation, tail_id = item99.head_id, item99, item99.tail_id
    assert type(head_id) == int and type(relation) == Sentence and type(tail_id) == int
    print(relation)


def test_fb13k_corpus_load():
    corpus = FB15K_237("./resources/dataset", "train_test.txt", "valid_test.txt", "test_test.txt")

    train = corpus.train
    dev = corpus.dev
    test = corpus.test

    assert train is not None and dev is not None and test is not None

    assert len(train) == 100
    assert len(dev) == 100
    assert len(test) == 100

    assert train.entity2id == dev.entity2id and dev.entity2id == test.entity2id


def test_dataloader():
    corpus = FB15K_237("./resources/dataset", "train_test.txt", "valid_test.txt", "test_test.txt")

    train = corpus.train
    dev = corpus.dev
    test = corpus.test

    train_loader = ColumnDataLoader(train, 10, sentence_level_batch=True)

    train_loader_iter = iter(train_loader)

    first_batch = next(train_loader_iter)
    assert len(first_batch) == 10
    assert type(first_batch[0]) == Sentence

    head, relation, tail = zip(*[(elm.head_id, elm, elm.tail_id) for elm in first_batch])
    assert len(head) == 10 and len(relation) == 10 and len(tail) == 10

    for batch in train_loader_iter:
        head, relation, tail = zip(*[(elm.head_id, elm, elm.tail_id) for elm in batch])
        assert len(head) == 10 and len(relation) == 10 and len(tail) == 10


def test_gettoken():
    corpus = FB15K_237("./resources/dataset", "train_test.txt", "valid_test.txt", "test_test.txt")

    tokens = corpus.get_train_full_tokenset(-1, -1)

    assert len(tokens) == 2

    assert len(tokens[0]) > 0 and len(tokens[1]) > 0
