from model import *
from utils import *
from main import validate
import copy
from torch.nn import CrossEntropyLoss


def load_best(best_model, test_loader):
    best_model.load_state_dict(torch.load('best_val_model.pt'))
    best_model = best_model.to(get_device())
    acc, _ = validate(best_model, test_loader, CrossEntropyLoss())
    print(f"Test accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    train_sents, train_tags = get_data('conll2003_train.pkl')
    val_sents, val_tags = get_data('conll2003_val.pkl')
    test_sents, test_tags = get_data('conll2003_test.pkl')

    corpus = copy.deepcopy(train_sents)
    corpus.extend(val_sents)
    corpus.extend(test_sents)

    test_data = DataNER(test_sents, test_tags, corpus)
    test_loader = DataLoader(test_data, batch_size=1, collate_fn=collate_fn)

    model = RNN(vocab_size=len(test_data.word2index))
    load_best(model, test_loader)