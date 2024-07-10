from datasets import load_dataset
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
    print("Preparing data...")
    try:
        ds = load_dataset(
            "ccmusic-database/pianos",
            name="eval",
            cache_dir="./__pycache__",
        )
        classes = ds["test"].features["label"].names
        use_hf = True

    except ConnectionError:
        ds = MsDataset.load(
            "ccmusic-database/pianos",
            subset_name="eval",
            cache_dir="./__pycache__",
        )
        classes = ds["test"]._hf_ds.features["label"].names
        use_hf = False

    if use_fl:
        num_samples_in_each_category = {k: 0 for k in classes}
        for item in ds["train"]:
            num_samples_in_each_category[classes[item["label"]]] += 1

        print("Data prepared.")
        return ds, classes, list(num_samples_in_each_category.values()), use_hf

    else:
        print("Data prepared.")
        return ds, classes, [], use_hf


def load_data(
    ds,
    input_size,
    use_hf,
    has_bn=False,
    batch_size=4,
    shuffle=True,
    num_workers=2,
):
    print("Loadeding data...")
    bs = batch_size
    ds_train = ds["train"]
    ds_valid = ds["validation"]
    ds_test = ds["test"]

    if not use_hf:
        ds_train = ds_train._hf_ds
        ds_valid = ds_valid._hf_ds
        ds_test = ds_test._hf_ds

    if has_bn:
        print("The model has bn layer")
        if bs < 2:
            print("Switch batch_size >= 2")
            bs = 2

    trainset = ds_train.with_transform(partial(transform, input_size=input_size))
    validset = ds_valid.with_transform(partial(transform, input_size=input_size))
    testset = ds_test.with_transform(partial(transform, input_size=input_size))

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
    print("Data loaded.")

    return traLoader, valLoader, tesLoader
