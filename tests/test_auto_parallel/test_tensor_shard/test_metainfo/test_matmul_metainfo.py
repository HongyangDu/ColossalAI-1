from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.node_handler import LinearModuleHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    MemoryCost,
    OperationData,
    OperationDataType,
    ShardingStrategy,
    StrategiesVector,
    TrainCycleItem,
)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.testing.utils import parameterize, rerun_if_address_is_in_use
from colossalai.utils import free_port
from tests.test_auto_parallel.test_tensor_shard.test_metainfo.utils import mem_test_for_node_strategy

if torch.__version__ >= '1.12.0':
    from colossalai.auto_parallel.meta_profiler import MetaInfo, meta_register


@pytest.mark.skipif(torch.__version__ < '1.12.0', reason="need pytorch 1.12.0 or higher for aten level operations")
@parameterize(
    'tensor_shapes',
    [
        [[128], [128]],    # dot product
        [[64, 128], [128]],    # mat-vec
        [[128], [128, 64]],    # vec-mat
        [[64, 64, 128], [128]],    # batched mat-vec
        [[128], [64, 128, 64]],    # vec-batched mat
        [[64, 128], [128, 192]],    # mat-mat
        [[64, 64, 128], [128, 192]],    # batched mat-mat
        [[64, 128], [64, 128, 192]],    # mat-batched mat
        [[64, 64, 128], [64, 128, 192]],    # batched mat-batched mat (matched batch dims)
        [[64, 1, 64, 128], [64, 128, 192]],    # batched mat-batched mat (unmatched batch dims)
    ])
def test_matmul_function_meta_info(tensor_shapes):
    meta_func = meta_register.get(torch.matmul)

    # construct meta tensors
    input_tensor = torch.rand(*tensor_shapes[0], device="meta")
    other_tensor = torch.rand(*tensor_shapes[1], device="meta")
    output_tensor = torch.matmul(input_tensor, other_tensor)

    # construct operation data
    input_data = OperationData(
        name="input",
        data=input_tensor,
        type=OperationDataType.ARG,
        logical_shape=input_tensor.shape,
    )
    other_data = OperationData(
        name="other",
        data=other_tensor,
        type=OperationDataType.ARG,
        logical_shape=other_tensor.shape,
    )
    output_data = OperationData(
        name="output",
        data=output_tensor,
        type=OperationDataType.OUTPUT,
        logical_shape=output_tensor.shape,
    )

    # construct args and kwargs
    args = [input_data, other_data, output_data]
    kwargs = {'inplace': False}

    # estimated results
    compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out = meta_func(*args, **kwargs)

    # actual results
    input_real_tensor = torch.rand(*tensor_shapes[0], device="cuda:0")
    other_real_tensor = torch.rand(*tensor_shapes[1], device="cuda:0")

    input_real_tensor.requires_grad = True
    other_real_tensor.requires_grad = True

    # fwd
    torch.cuda.reset_peak_memory_stats()
    mem_stamp0 = torch.cuda.memory_allocated()
    output_real_tensor = torch.matmul(input_real_tensor, other_real_tensor)
    fwd_allocated = torch.cuda.memory_allocated() - mem_stamp0
    fwd_peak = torch.cuda.max_memory_allocated() - mem_stamp0

    # bwd
    upstream_grad = torch.rand_like(output_real_tensor)
    torch.cuda.reset_peak_memory_stats()
    mem_stamp0 = torch.cuda.memory_allocated()
    torch.autograd.backward(output_real_tensor, upstream_grad)
    bwd_allocated = torch.cuda.memory_allocated() - mem_stamp0
    bwd_peak = torch.cuda.max_memory_allocated() - mem_stamp0

    compute_cost: TrainCycleItem
    memory_cost: TrainCycleItem

    print("=====================")
    print(f"input shapes: {tensor_shapes[0]}, {tensor_shapes[1]}")
    print(f"output shapes: {output_tensor.shape}")

    # estimated results
    print("Estimated Results")

    # compute cost
    print("compute_cost:")
    print(f"    fwd: {compute_cost.fwd}")
    print(f"    bwd: {compute_cost.bwd}")

    # memory cost
    print("memory_cost:")
    # fwd
    print(f"    fwd activation: {memory_cost.fwd.activation / 1024} KB")
    print(f"    fwd buffer: {memory_cost.fwd.buffer / 1024} KB")
    print(f"    fwd temp: {memory_cost.fwd.temp / 1024} KB")
    print(f"    fwd parameter: {memory_cost.fwd.parameter / 1024} KB")

    # bwd
    print(f"    bwd activation: {memory_cost.bwd.activation / 1024} KB")
    print(f"    bwd buffer: {memory_cost.bwd.buffer / 1024} KB")
    print(f"    bwd temp: {memory_cost.bwd.temp / 1024} KB")
    print(f"    bwd parameter: {memory_cost.bwd.parameter / 1024} KB")

    # actual results
    print("Actual Results")

    print("memory_cost:")
    # fwd
    print(f"    fwd allocated: {fwd_allocated / 1024} KB")
    print(f"    fwd peak: {fwd_peak / 1024} KB")

    # bwd
    print(f"    bwd allocated: {bwd_allocated / 1024} KB")
    print(f"    bwd peak: {bwd_peak / 1024} KB")


if __name__ == '__main__':
    test_matmul_function_meta_info()
