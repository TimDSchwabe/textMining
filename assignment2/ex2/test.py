from model import *
from utils import *
from main import validate

def load_best(best_model, test_loader):
    best_model.load_state_dict(torch.load('best_val_model.pt'))
    print(f"Test accuracy: {validate(best_model, test_loader)}")

if __name__ == "__main__":
    train_sents, train_tags = get_data('conll2003_train.pkl')
    val_sents, val_tags = get_data('conll2003_val.pkl')
    test_sents, test_tags = get_data('conll2003_test.pkl')

    sents_cpy = copy.deepcopy(train_sents)
    sents_cpy.extend(val_sents)
    sents_cpy.extend(test_sents)

    test_data = DataNER(test_sents, test_tags, sents_cpy)
    test_loader = DataLoader(test_data, batch_size=4)

    model = RNN(vocab_size=len(test_data.word2index))
    load_best(model, test_loader)