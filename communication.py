import config


def get_allreduce_cost(tensor_size_bytes: int) -> float:
    per_hop_data_size = tensor_size_bytes / config.TP_DEGREE  # Byte
    time_per_hop_us = (
        per_hop_data_size / (config.NVLINK_LINK_NUM * config.NVLINK_BW * 1e3)
    ) + config.NVLINK_LATENCY
    hop_count = (config.TP_DEGREE - 1) * 2
    total_time_us = time_per_hop_us * hop_count
    return total_time_us


# #NVLINK
# def get_alltoall_cost(ep_degree: int, tensor_size_bytes: int) -> float:

#     send_data_size = (tensor_size_bytes/ep_degree) * 8 / config.TP_DEGREE
#     count = (ep_degree - 1) / (config.NVLINK_LINK_NUM/2)
#     send_time = (config.NVLINK_LATENCY + (send_data_size/(config.NVLINK_BW*1e3)))
#     total_time_us = send_time * count
#     return total_time_us

# #INFINI BAND - Ring
# def get_alltoall_cost(ep_degree: int, tensor_size_bytes: int) -> float:
#     send_data_size = ((tensor_size_bytes - (tensor_size_bytes/ep_degree)))/config.TP_DEGREE
#     count = ep_degree - 1
#     send_time = (config.INFINI_LATENCY + (send_data_size/(config.INFINI_BW*1e3)))
#     total_time_us = send_time * count
#     return total_time_us


# INFINI BAND
def get_moe_scatter_cost(tensor_size_bytes: int) -> float:
    ep_degree = config.TP_DEGREE * config.DP_DEGREE
    send_data_size = (
        (tensor_size_bytes * 8) / config.TP_DEGREE * ((ep_degree - config.TP_DEGREE) / ep_degree)
    )
    count = 1
    send_time = config.INFINI_LATENCY + (send_data_size / (config.INFINI_BW * 1e3))
    total_time_us = send_time * count
    return total_time_us


# #NVLINK
# def get_moe_allreduce_cost(ep_degree: int, tensor_size_bytes: int) -> float:

#     send_data_size = (tensor_size_bytes/ep_degree) * 8
#     count = (ep_degree - 1) / (config.NVLINK_LINK_NUM/2)
#     send_time = (config.NVLINK_LATENCY + (send_data_size/(config.NVLINK_BW*1e3)))
#     total_time_us = send_time * count
#     return total_time_us

# # #INFINI BAND - Ring
# def get_moe_allreduce_cost(ep_degree: int, tensor_size_bytes: int) -> float:
#     send_data_size = (tensor_size_bytes - (tensor_size_bytes/ep_degree))/2
#     count = ep_degree - 1
#     send_time = (config.INFINI_LATENCY + (send_data_size/(config.INFINI_BW*1e3)))
#     total_time_us = send_time * count
#     return total_time_us


# Infiniband for inter node, Nvlink for intranode
def get_moe_gather_cost(tensor_size_bytes: int) -> float:
    ep_degree = config.TP_DEGREE * config.DP_DEGREE
    nvlink_bw = config.NVLINK_BW * config.NVLINK_LINK_NUM
    internode_receive_size = (
        (tensor_size_bytes * 8) / config.TP_DEGREE * ((ep_degree - config.TP_DEGREE) / ep_degree)
    )
    intranode_receive_size = (
        (tensor_size_bytes * 8) * (config.TP_DEGREE / ep_degree)
    )
    count = 1
    receive_time = max(
        config.INFINI_LATENCY + (internode_receive_size / (config.INFINI_BW * 1e3)),
        config.NVLINK_LATENCY + (intranode_receive_size / (config.NVLINK_BW * 1e3)),
    )
    total_time_us = receive_time * count
    return total_time_us
