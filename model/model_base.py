from abc import abstractmethod
from typing import List
import torch
from torch import nn
from data.dataset import MyDataPoint, MyPair


class MyBaseModel(nn.Module):
    @staticmethod
    def split(data_points, batch_size=8):
        num_batch = (len(data_points) + batch_size - 1) // batch_size
        return [data_points[i * batch_size:(i + 1) * batch_size] for i in range(num_batch)]

    @staticmethod
    def use_cache(module: nn.Module, data_points: List[MyDataPoint]):
        for parameter in module.parameters():
            if parameter.requires_grad:
                return False
        for data_point in data_points:
            if data_point.embedding is None:
                return False
        return True

    @abstractmethod
    def encode(self, pairs: List[MyPair], fuse: str = ''):
        pass

    def forward(self, pairs: List[MyPair]):
        if isinstance(pairs, MyPair):
            pairs = [pairs]
        feats = self.encode(pairs)
        feats = self.dropout(feats)
        logits = self.classifier(feats)
        return logits

    def predict(self, data_loader):
        pred_flags = []
        self.eval()
        with torch.no_grad():
            for pairs in data_loader:
                logits = self.forward(pairs)
                pred_flags += torch.argmax(logits, dim=1).tolist()
        return pred_flags
