# Local reconstruction of the NVIDIA DLI run_assessment helper.
#
# The official file ships on NVIDIA's assessment server at
# /dli/assessment/notebook_helpers/run_assessment.py and is NOT included
# in this Brev deployment (the instance only has the tutorial
# environment). This version reproduces the official file's documented
# behaviour, as shown in NVIDIA's own forum traces and the notebook
# intro: evaluate the model on the validation set and report accuracy
# against the 0.92 pass mark. It is a self-check, not the official
# grader. The certificate itself comes from the course page's
# "ASSESS TASK" button.
#
# Install: place this file in /dli/tutorials/ (the notebook's folder),
# e.g. by uploading it through the JupyterLab file browser, then run
# the notebook's final cells:
#     from run_assessment import run_assessment
#     run_assessment(my_model)

import glob

import torch
import torchvision.io as tv_io
from torch.utils.data import DataLoader, Dataset
from torchvision.models import VGG16_Weights

DATA_LABELS = ["freshapples", "freshbanana", "freshoranges",
               "rottenapples", "rottenbanana", "rottenoranges"]
VALID_PATH = "data/fruits/valid/"
BATCH_SIZE = 32


class _ValidDataset(Dataset):
    """Mirror of the notebook's MyDataset, validation split only."""

    def __init__(self, data_dir, transform, device):
        self.imgs = []
        self.labels = []
        for l_idx, label in enumerate(DATA_LABELS):
            for path in glob.glob(data_dir + label + "/*.png", recursive=True):
                img = tv_io.read_image(path, tv_io.ImageReadMode.RGB)
                self.imgs.append(transform(img).to(device))
                self.labels.append(torch.tensor(l_idx).to(device))

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

    def __len__(self):
        return len(self.imgs)


def _valid_loader(device):
    weights = VGG16_Weights.DEFAULT
    pre_trans = weights.transforms()
    dataset = _ValidDataset(VALID_PATH, pre_trans, device)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def validate(model):
    device = next(model.parameters()).device
    loader = _valid_loader(device)
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            accuracy += (predicted == y).sum().item()
    return accuracy / len(loader.dataset)


def run_assessment(model):
    print("Evaluating model to obtain average accuracy...\n")
    average = validate(model)
    print("\nAccuracy required to pass the assessment is 0.92 or greater.")
    print("Your average accuracy is {:5.4f}.\n".format(average))
    if average >= 0.92:
        print("Congratulations! You passed the assessment.")
    else:
        print("Your accuracy is not yet high enough to pass the assessment, please continue trying.")
