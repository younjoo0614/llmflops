import config


#outproj용 all reduce
def get_allreduce_cost(tp_degree: int, tensor_size_bytes: int) -> float:

    size_per_hop = tensor_size_bytes  # Byte
    time_per_hop_us = (size_per_hop /(config.NVLINK_LINK_NUM*config.NVLINK_BW*1e3)) + config.NVLINK_LATENCY
    hop_count = tp_degree - 1 #ring all reduce로 생각

    total_time_us = time_per_hop_us * hop_count
    return (total_time_us)


# #NVLINK
# def get_alltoall_cost(ep_degree: int, tensor_size_bytes: int) -> float:

#     send_data_size = (tensor_size_bytes/ep_degree) * 8 / config.TP_DEGREE 
#     count = (ep_degree - 1) / (config.NVLINK_LINK_NUM/2) 
#     send_time = (config.NVLINK_LATENCY + (send_data_size/(config.NVLINK_BW*1e3)))
#     total_time_us = send_time * count
#     return total_time_us

#INFINI BAND
def get_alltoall_cost(ep_degree: int, tensor_size_bytes: int) -> float:
    send_data_size = ((tensor_size_bytes - (tensor_size_bytes/ep_degree)))/config.TP_DEGREE
    count = ep_degree - 1
    send_time = (config.INFINI_LATENCY + (send_data_size/(config.INFINI_BW*1e3)))
    total_time_us = send_time * count
    return total_time_us



# #NVLINK
# def get_moe_allreduce_cost(ep_degree: int, tensor_size_bytes: int) -> float:

#     send_data_size = (tensor_size_bytes/ep_degree) * 8
#     count = (ep_degree - 1) / (config.NVLINK_LINK_NUM/2)
#     send_time = (config.NVLINK_LATENCY + (send_data_size/(config.NVLINK_BW*1e3)))
#     total_time_us = send_time * count
#     return total_time_us

#INFINI BAND
def get_moe_allreduce_cost(ep_degree: int, tensor_size_bytes: int) -> float:
    send_data_size = (tensor_size_bytes - (tensor_size_bytes/ep_degree))/2
    count = ep_degree - 1
    send_time = (config.INFINI_LATENCY + (send_data_size/(config.INFINI_BW*1e3)))
    total_time_us = send_time * count
    return total_time_us
