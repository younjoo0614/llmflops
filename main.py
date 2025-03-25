import argparse
import pandas as pd
import config
from config import load_model_config, load_device_config
from matrix import Matrix
from layer import Layer
from model import Model
from graph import create_time_graph

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
    parser.add_argument("--data-size", type=int, help="Data Size", default=2)
    parser.add_argument("--model-num", type=int, help="Model Num", default=0)
    parser.add_argument("--flash-attention", action="store_true", default=False)
    parser.add_argument("--flash-mla", action="store_true", default=False)
    args = parser.parse_args()

    model_config = load_model_config(args.model_num)
    device_config = load_device_config()

    print("===== Configuration =====")
    print(f"Input Sequence Length: {args.input_len}")
    print(f"Output Sequence Length: {args.output_len}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Batch Size per Device: {args.batch_size / config.DP_DEGREE}")
    print(f"Data Size: {args.data_size}")
    print(f"Model Num: {args.model_num}")
    print(f"Model Name: {model_config['Model Name']}")
    print(f"Flash Attention: {args.flash_attention}\n\n")

    deepseek = Model("deepseek")

    deepseek_base = pd.DataFrame(columns=[
        "Layer Name", "FLOPS", "InputA", "InputB", "Output", "OP/B", "Execution_time"
    ])

    batch_size_per_device = int(args.batch_size / config.DP_DEGREE)

    deepseek.base_layer("deepseek", deepseek_base, args.input_len, args.output_len, batch_size_per_device,
                        config.TP_DEGREE, config.DP_DEGREE, model_config, False, False, fa_flag=args.flash_attention)
    deepseek_base.to_csv("./result/base/deepseek_base_prefill_dense.csv", index=False, encoding="utf-8")
    create_time_graph(deepseek_base, "/base/deepseek_base_prefill_dense",    input_len=args.input_len,
    output_len=args.output_len,
    batch_size=args.batch_size,
    tensor_parallel=config.TP_DEGREE,
    data_parallel=config.DP_DEGREE)

    deepseek_base_prefill_moe = pd.DataFrame(columns=[
        "Layer Name", "FLOPS", "InputA", "InputB", "Output", "OP/B", "Execution_time"
    ])
    deepseek.base_layer("deepseek", deepseek_base_prefill_moe, args.input_len, args.output_len, batch_size_per_device,
                        config.TP_DEGREE, config.DP_DEGREE, model_config, False, True, fa_flag=args.flash_attention)
    deepseek_base_prefill_moe.to_csv("./result/base/deepseek_base_prefill_moe.csv", index=False, encoding="utf-8")
    create_time_graph(deepseek_base_prefill_moe, "/base/deepseek_base_prefill_moe",    input_len=args.input_len,
    output_len=args.output_len,
    batch_size=args.batch_size,
    tensor_parallel=config.TP_DEGREE,
    data_parallel=config.DP_DEGREE)

    deepseek_base_decode = pd.DataFrame(columns=[
        "Layer Name", "FLOPS", "InputA", "InputB", "Output", "OP/B", "Execution_time"
    ])
    deepseek.base_layer("deepseek", deepseek_base_decode, args.input_len, args.output_len, batch_size_per_device,
                        config.TP_DEGREE, config.DP_DEGREE, model_config, True, False, fa_flag=args.flash_attention)
    deepseek_base_decode.to_csv("./result/base/deepseek_base_decode_dense.csv", index=False, encoding="utf-8")
    create_time_graph(deepseek_base_decode, "/base/deepseek_base_decode_dense",    input_len=args.input_len,
    output_len=args.output_len,
    batch_size=args.batch_size,
    tensor_parallel=config.TP_DEGREE,
    data_parallel=config.DP_DEGREE)

    deepseek_base_decode_moe = pd.DataFrame(columns=[
        "Layer Name", "FLOPS", "InputA", "InputB", "Output", "OP/B", "Execution_time"
    ])
    deepseek.base_layer("deepseek", deepseek_base_decode_moe, args.input_len, args.output_len, batch_size_per_device,
                        config.TP_DEGREE, config.DP_DEGREE, model_config, True, True, fa_flag=args.flash_attention)
    deepseek_base_decode_moe.to_csv("./result/base/deepseek_base_decode_moe.csv", index=False, encoding="utf-8")
    create_time_graph(deepseek_base_decode_moe, "/base/deepseek_base_decode_moe",    input_len=args.input_len,
    output_len=args.output_len,
    batch_size=args.batch_size,
    tensor_parallel=config.TP_DEGREE,
    data_parallel=config.DP_DEGREE)

    df_w_uk_first = pd.DataFrame(columns=[
        "Layer Name", "FLOPS", "InputA", "InputB", "Output", "OP/B", "Execution_time"
    ])
    deepseek.w_uk_first_layer("deepseek", df_w_uk_first, args.input_len, args.output_len,
                               batch_size_per_device, config.TP_DEGREE, config.DP_DEGREE, model_config, False, False)
    df_w_uk_first.to_csv("./result/absorb/deepseek_w_uk_first_prefill_dense.csv", index=False, encoding="utf-8")
    create_time_graph(df_w_uk_first, "/absorb/deepseek_w_uk_first_prefill_dense",input_len=args.input_len,
    output_len=args.output_len,
    batch_size=args.batch_size,
    tensor_parallel=config.TP_DEGREE,
    data_parallel=config.DP_DEGREE)

    df_w_uk_first_prefill_moe = pd.DataFrame(columns=[
        "Layer Name", "FLOPS", "InputA", "InputB", "Output", "OP/B", "Execution_time"
    ])
    deepseek.w_uk_first_layer("deepseek", df_w_uk_first_prefill_moe, args.input_len, args.output_len,
                              batch_size_per_device, config.TP_DEGREE, config.DP_DEGREE, model_config, False, True)
    df_w_uk_first_prefill_moe.to_csv("./result/absorb/deepseek_w_uk_first_prefill_moe.csv", index=False, encoding="utf-8")
    create_time_graph(df_w_uk_first_prefill_moe, "/absorb/df_w_uk_first_prefill_moe",    input_len=args.input_len,
    output_len=args.output_len,
    batch_size=args.batch_size,
    tensor_parallel=config.TP_DEGREE,
    data_parallel=config.DP_DEGREE)

    df_w_uk_first_decode = pd.DataFrame(columns=[
        "Layer Name", "FLOPS", "InputA", "InputB", "Output", "OP/B", "Execution_time"
    ])
    deepseek.w_uk_first_layer("deepseek", df_w_uk_first_decode, args.input_len, args.output_len,
                              batch_size_per_device, config.TP_DEGREE, config.DP_DEGREE, model_config, True, False)
    df_w_uk_first_decode.to_csv("./result/absorb/deepseek_w_uk_first_decode_dense.csv", index=False, encoding="utf-8")
    create_time_graph(df_w_uk_first_decode, "/absorb/deepseek_w_uk_first_decode_dense",    input_len=args.input_len,
    output_len=args.output_len,
    batch_size=args.batch_size,
    tensor_parallel=config.TP_DEGREE,
    data_parallel=config.DP_DEGREE)

    df_w_uk_first_decode_moe = pd.DataFrame(columns=[
        "Layer Name", "FLOPS", "InputA", "InputB", "Output", "OP/B", "Execution_time"
    ])
    deepseek.w_uk_first_layer("deepseek", df_w_uk_first_decode_moe, args.input_len, args.output_len,
                              batch_size_per_device, config.TP_DEGREE, config.DP_DEGREE, model_config, True, True)
    df_w_uk_first_decode_moe.to_csv("./result/absorb/deepseek_w_uk_first_decode_moe.csv", index=False, encoding="utf-8")
    create_time_graph(df_w_uk_first_decode_moe, "/absorb/deepseek_w_uk_first_decode_moe",    input_len=args.input_len,
    output_len=args.output_len,
    batch_size=args.batch_size,
    tensor_parallel=config.TP_DEGREE,
    data_parallel=config.DP_DEGREE)

if __name__ == "__main__":
    main()
