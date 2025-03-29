import config


def get_allreduce_cost(tp_degree: int, tensor_size_bytes: int) -> float:

    if tp_degree <= 1:
        return 0.0

    size_per_hop = tensor_size_bytes / tp_degree  # 바이트
    time_per_hop_us = (size_per_hop / config.NVLINK_BW) / 1e3 + config.NVLINK_LATENCY
    hop_count = (tp_degree - 1) * 2

    total_time_us = time_per_hop_us * hop_count
    return (total_time_us)