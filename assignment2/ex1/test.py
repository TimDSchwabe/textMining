import torch
from torch.utils.data import DataLoader
from utils import WinogradDataset
import utils
from model import ClassifierWithAttention
import json


def load_model():
    settings = {}
    with open("model_settings.json", "r") as f:
        settings = json.load(f)
    model = ClassifierWithAttention(**settings)
    try:
        model.load_state_dict(torch.load("model.pt"))
    except:
        raise Exception("model loading failed")
    model = model.to(utils.get_device())
    model.eval()
    return model

# it is notable that models I tested always predicted the same label, so I am not sure if the model is actually working
test_dataset = WinogradDataset("WNLI/test.tsv")
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=utils.collate)
model = load_model()
with torch.no_grad():
    for batch in test_loader:
        sent1, sent2, label = batch
        output = model(sent1, sent2).squeeze()
        acc = utils.accuracy(output, label.to(utils.get_device()))
        print(f"Test accuracy: {acc*100:.2f}%")
