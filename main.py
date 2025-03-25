import argparse
import pandas as pd
import config
from config import load_model_config, load_device_config, set_parallelism_degree
from matrix import Matrix
from layer import Layer
from model import Model

def print_layer_config(layer_config, indent=0):
    for key, value in layer_config.items():
        print("  " * indent + f"- {key}")
        if isinstance(value, dict):
            print_layer_config(value, indent + 1)
        elif isinstance(value, list):
            for item in value:
                print("  " * (indent + 1) + f"- {item}")
        else:
            print("  " * (indent + 1) + f"  {value}")

def main():
    parser = argparse.ArgumentParser(description="Process model training with configurations")
    parser.add_argument("--input-len", type=int, required=True, help="Input Sequence Length")
    parser.add_argument("--output-len", type=int, required=True, help="Output Sequence Length")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch Size")
    parser.add_argument("--tensor-degree", type=int, required=True, help="Tensor parallelism degree")
    parser.add_argument("--data-degree", type=int, required=True, help="Data parallelism degree")
    parser.add_argument("--data-size", type=int, required=True, help="Data Size")
    parser.add_argument("--model-num", type=int, required=True, help="Model Num")
    parser.add_argument("--flash-attention", action="store_true", default=False)
    parser.add_argument("--flash-mla", action="store_true", default=False)
    args = parser.parse_args()

    model_config = load_model_config(args.model_num)
    device_config = load_device_config()
    
    set_parallelism_degree(int(args.tensor_degree), int(args.data_degree))

    print("===== Configuration =====")
    print(f"Input Sequence Length: {args.input_len}")
    print(f"Output Sequence Length: {args.output_len}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Batch Size per Device: {args.batch_size / args.data_degree}")
    print(f"Data Size: {args.data_size}")
    print(f"Model Num: {args.model_num}")
    print(f"Model Name: {model_config['Model Name']}")
    print(f"Flash Attention: {args.flash_attention}\n\n")

    deepseek = Model("deepseek")

    batch_size_per_dp_node = int(args.batch_size / args.data_degree)
    
    for impl in ["base", "absorb"]:
        for decode_flag in [True, False]:
            for moe_flag in [True, False]:
                
                csv_data = pd.DataFrame(columns=["Layer Name", "FLOPS", "InputA", "InputB", "Output", "OP/B", "Execution_time"])
                if impl == "base":
                    deepseek.base_layer("deepseek", csv_data, args.input_len,args.output_len, batch_size_per_dp_node, args.data_size,
                        args.tensor_degree, args.data_degree, model_config, decode_flag, moe_flag, args.flash_attention)
                elif impl == "absorb":
                    deepseek.w_uk_first_layer("deepseek", csv_data, args.input_len,args.output_len, batch_size_per_dp_node, args.data_size, args.tensor_degree, args.data_degree, model_config, decode_flag, moe_flag)
                
                csv_name = "./result/deepseek_{}_{}_{}_Lin{}_Lout{}_Batch{}_TP{}_DP{}".format("decode" if decode_flag else "prefill", impl, "moe" if moe_flag else "dense", args.input_len, args.output_len, args.batch_size, config.TP_DEGREE, config.DP_DEGREE)
                csv_data.to_csv(csv_name+".csv", index=False, encoding="utf-8")
                


if __name__ == "__main__":
    main()
