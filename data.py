from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from modelscope.msdatasets import MsDataset
from torchvision.transforms import *


def transform(example_batch, input_size=300):
    compose = Compose(
        [
            Resize([input_size, input_size]),
            RandomAffine(5),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    inputs = [compose(x.convert("RGB")) for x in example_batch["mel"]]
    example_batch["mel"] = inputs
    return example_batch


def prepare_data(use_fl: bool):
    print("Preparing & loading data...")
    ds = MsDataset.load(
        "ccmusic-database/pianos",
        subset_name="eval",
        cache_dir="./__pycache__",
    )
    classes = ds["test"].features["label"].names
    sizes = []
    if use_fl:
        num_samples_in_each_category = {k: 0 for k in classes}
        for item in tqdm(ds["train"], desc="Statistics by category for focal loss..."):
            num_samples_in_each_category[classes[item["label"]]] += 1

        sizes = list(num_samples_in_each_category.values())

    return ds, classes, sizes


def load_data(
    ds: MsDataset,
    insize,
    has_bn=False,
    batch_size=4,
    shuffle=True,
    num_workers=2,
):
    bs = batch_size
    if has_bn:
        print("The model has bn layer")
        if bs < 2:
            print("Switch batch_size >= 2")
            bs = 2

    trainset = ds["train"].with_transform(partial(transform, input_size=insize))
    validset = ds["validation"].with_transform(partial(transform, input_size=insize))
    testset = ds["test"].with_transform(partial(transform, input_size=insize))

    traLoader = DataLoader(
        trainset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=has_bn,
    )
    valLoader = DataLoader(
        validset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=has_bn,
    )
    tesLoader = DataLoader(
        testset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=has_bn,
    )

    return traLoader, valLoader, tesLoader
