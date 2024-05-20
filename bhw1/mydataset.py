import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, idx: np.array, test: bool, labels: pd.DataFrame, transform):
        super().__init__()
        self.transform = transform
        self.numbers_of_set = idx
        self.is_test = test
        self.labels = labels

    def __len__(self):
        return self.numbers_of_set.shape[0]

    def __getitem__(self, item):
        if self.is_test:
            image_path = "test/test_" + f"{str(self.numbers_of_set[item]).zfill(5)}" + ".jpg"
        else:
            image_path = "trainval/trainval_" + f"{str(self.numbers_of_set[item]).zfill(5)}" + ".jpg"
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        if self.is_test:
            return image, np.zeros(0)
        label = self.labels.iloc[[self.numbers_of_set[item]], :]['Category']
        return image, np.array(label)