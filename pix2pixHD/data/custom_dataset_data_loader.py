import torch.utils.data
from data.base_data_loader import BaseDataLoader
import random
import functools

def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

def collate(dataset, batch):
    # recursively checks if the image numbers are in order, if not calls itself on a new batch
    numbers = [data['number'] for data in batch]
    if all(numbers[i] + 1 == numbers[i + 1] for i in range(len(numbers) - 1)):
        return torch.utils.data.dataloader.default_collate(batch)

    else:
        start = random.randint(0, len(dataset) -1 - len(batch))
        replacement_batch = [dataset[start + i] for i in range(len(batch))]
        return collate(dataset, replacement_batch)

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.collate_fn = functools.partial(collate, self.dataset)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            # collate_fn=self.collate_fn,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
