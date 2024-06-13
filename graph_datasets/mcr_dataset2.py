import os.path as osp
import shutil
from typing import Callable, List, Optional
import torch
import torch_geometric

from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.data import Data
import fs


class MCRDataset(InMemoryDataset):
    url = ('https://github.com/pinkocto/DataHub/raw/main/'
                   'graph_datasets/datasets')
    
    cleaned_url = ('https://github.com/pinkocto/DataHub/raw/main/'
                   'graph_datasets/datasets')

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        use_node_attr: bool = True,
        use_edge_attr: bool = False,
        cleaned: bool = True,
    ) -> None:
        self.name = name
        self.cleaned = cleaned
        self.data = None  # data를 초기화합니다.
        super().__init__(root, transform, pre_transform, pre_filter)

        out = torch.load(self.processed_paths[0])
        if not isinstance(out, tuple) or len(out) < 3:
            raise RuntimeError(
                
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:  # Backward compatibility.
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        assert isinstance(self.data, Data)
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attrs = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attrs:]

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_labels(self) -> int:
        return self.sizes['num_edge_labels']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        url = self.cleaned_url if self.cleaned else self.url
        fs.cp(f'{url}/{self.name}.zip', self.raw_dir, extract=True)
        for filename in fs.ls(osp.join(self.raw_dir, self.name)):
            fs.mv(filename, osp.join(self.raw_dir, osp.basename(filename)))
        fs.rm(osp.join(self.raw_dir, self.name))

    def process(self) -> None:
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        assert isinstance(self.data, Data)
        torch.save(
            (self.data.to_dict(), self.slices, sizes, self.data.__class__),
            self.processed_paths[0],
        )

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
