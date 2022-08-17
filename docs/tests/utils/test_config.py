import pytest

from wrangler.utils.config import load_dataset_object
from wrangler.data.datasets import (
    AbstractDataset
)

def test_load_dataset_object():
    ds_obj = load_dataset_object(obj_name='MemoryDataset',obj_path="wrangler.data.datasets")
    ds = ds_obj(data=10)
    assert isinstance(ds, AbstractDataset)


