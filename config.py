import json
import pandas as pd

def load_model_config(model_num, model_config_file="./model_config.json"):
    with open(model_config_file, mode="r", encoding="utf-8") as file:
        model_config = json.load(file)
    for model in model_config["models"]:
        if model["Model Num"] == model_num:
            global NUM_HEADS
            NUM_HEADS = int(model['n_heads'])
            return model
    return None

def load_device_config(device_config_file="./device_config.json"):
    with open(device_config_file, mode="r", encoding="utf-8") as file:
        device_config = json.load(file)
        global TP_DEGREE, DP_DEGREE, SH_MEM
        TP_DEGREE = int(device_config["TP_DEGREE"])
        DP_DEGREE = int(device_config["DP_DEGREE"])
        SH_MEM = int(device_config["SH_MEM"])


def create_layer_dataframe(layer_config):
    columns = [
        "Layer Name",
        "Input",
        "Weight",
        "Output",
        "FLOPs", "FLOPs/bytes"
    ]
    
    data = []

    def extract_layers(config, prefix):
        for layer in config["Attention block"]:
            data.append([f"{prefix} > {layer}"] + [0] * (len(columns) - 1))
        
        if isinstance(config["FeedForward block"], list):
            for layer in config["FeedForward block"]:
                data.append([f"{prefix} > {layer}"] + [0] * (len(columns) - 1))
        elif isinstance(config["FeedForward block"], dict):  # MoE의 Gate 처리
            for gate_layer in config["FeedForward block"]:
                data.append([f"{prefix} > {gate_layer}"] + [0] * (len(columns) - 1))

    extract_layers(layer_config["Dense"], "Prefill Dense")
    extract_layers(layer_config["MoE"], "Prefill MoE")
    extract_layers(layer_config["Dense"], "Decode Dense")
    extract_layers(layer_config["MoE"], "Decode MoE")

    df = pd.DataFrame(data, columns=columns)
    return df
