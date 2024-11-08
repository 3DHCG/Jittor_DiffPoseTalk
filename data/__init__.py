from .datasets import LmdbDatasetForSE, LmdbDataset


def infinite_data_loader(data_loader):
    while True:
        for data in data_loader:
            yield data
