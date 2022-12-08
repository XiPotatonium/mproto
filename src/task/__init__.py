from typing import Any, Dict, List
import numpy as np
import torch
from torch import Tensor

from alchemy.registry import Registrable


class CollateFn(Registrable):
    @classmethod
    def from_registry(cls, ty: str, **kwargs):
        """参数中不要包含ctx等不方便跨进程的东西

        Args:
            ty (str): _description_

        Returns:
            _type_: _description_
        """
        handler_cls = cls.resolve_registered_module(ty)
        handler = handler_cls(**kwargs)
        return handler


@CollateFn.register("Default")
class DefaultCollateFn(CollateFn):
    def __init__(self, **kwargs) -> None:
        super().__init__()
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {}

        for key in batch[0].keys():
            data = []
            for sample in batch:
                data_item = sample[key]
                if isinstance(data_item, np.ndarray):
                    data_item = torch.from_numpy(data_item)
                data.append(data_item)

            # TODO: better implementation
            if isinstance(data[0], (Tensor)):
                ret[key] = DefaultCollateFn.padded_stack_tensor(data)
            else:
                ret[key] = data

        return ret

    @staticmethod
    def padded_stack_tensor(tensors: List[Tensor], padding=0):
        dim_count = len(tensors[0].shape)

        max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
        padded_tensors = []

        for t in tensors:
            e = DefaultCollateFn.extend_tensor(t, max_shape, fill=padding)
            padded_tensors.append(e)

        stacked = torch.stack(padded_tensors)
        return stacked
    
    @staticmethod
    def extend_tensor(tensor, extended_shape, fill=0):
        tensor_shape = tensor.shape

        extended_tensor = torch.zeros(
            extended_shape, dtype=tensor.dtype).to(tensor.device)
        extended_tensor = extended_tensor.fill_(fill)

        if len(tensor_shape) == 1:
            extended_tensor[:tensor_shape[0]] = tensor
        elif len(tensor_shape) == 2:
            extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
        elif len(tensor_shape) == 3:
            extended_tensor[:tensor_shape[0],
            :tensor_shape[1], :tensor_shape[2]] = tensor
        elif len(tensor_shape) == 4:
            extended_tensor[:tensor_shape[0], :tensor_shape[1],
            :tensor_shape[2], :tensor_shape[3]] = tensor

        return extended_tensor
