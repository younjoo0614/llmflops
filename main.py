import argparse
import json
import config
from config import load_model_config, create_layer_dataframe
from matrix import Matrix
from layer import Layer
from model import Model
from graph import create_time_graph
import pandas as pd


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
    parser = argparse.ArgumentParser(
        description="Process model training with configurations")
    parser.add_argument("--input-len",type=int, required=True, help="Input Sequence Length")
    parser.add_argument("--output-len", type=int, required=True, help="Output Sequence Length")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch Size")
    parser.add_argument("--data-size", type=int, required=True, help="Data Size")
    parser.add_argument("--model-num", type=int, required=True, help="Model Num")
    parser.add_argument("--tp-degree", type=int, default=1)
    parser.add_argument("--dp-degree", type=int, default=1)

    args = parser.parse_args()
    model_config = load_model_config(args.model_num)

    print("===== Configuration =====")
    print(f"Input Sequence Length: {args.input_len}")
    print(f"Output Sequence Length: {args.output_len}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Data Size: {args.data_size}")
    print(f"Model Num: {args.model_num}")
    print(f"Model Name: {model_config['Model Name']}\n\n")
    print(f"deepseek_base prefiil dense\n")

    deepseek = Model("deepseek")

    deepseek_base = pd.DataFrame(columns=[
        "Layer Name", "FLOPS", "InputA", "InputB", "Output", "FLOPS", "OP/B", "Execution_time"
    ])
    deepseek.base_layer("deepseek", deepseek_base, args.input_len, args.output_len, args.batch_size, 
                        args.tensor_parallelism_degree, model_config, False, False)
    deepseek_base.to_csv("./result/base/deepseek_base_prefill_dense.csv", index=False, encoding="utf-8")

    create_time_graph(deepseek_base, "/base/deepseek_base_prefill_dense")


    # deepseek_base_prefill_moe = pd.DataFrame(columns=[
    #     "Layer Name", "FLOPS", "InputA", "InputB", "Output", "FLOPS", "OP/B",
    #     "Execution_time"
    # ])
    # print(f"\n\ndeepseek_w_uk_first prefill moe\n")
    # deepseek.base_layer("deepseek", deepseek_base_prefill_moe, args.input_len,
    #                     args.output_len, args.batch_size, model_config, False,
    #                     True)
    # print(deepseek_base_prefill_moe)
    # deepseek_base_prefill_moe.to_csv("./result/deepseek_base_prefill_moe.csv",
    #                                  index=False,
    #                                  encoding="utf-8")

    # # deepseek_base_decode = pd.DataFrame(columns=["Layer Name", "FLOPS", "InputA", "InputB", "Output"])
    # # print(f"\n\ndeepseek_w_uk_first decode dense\n")
    # # deepseek.base_layer("deepseek", deepseek_base_decode, input_seq_len, output_seq_len, batch_size, model_config, True, False)
    # # print(deepseek_base_decode)
    # # deepseek_base_decode.to_csv("./result/deepseek_base_decode_dense.csv", index=False, encoding="utf-8")
    # deepseek_base_decode = pd.DataFrame(columns=[
    #     "Layer Name", "FLOPS", "InputA", "InputB", "Output", "FLOPS", "OP/B",
    #     "Execution_time"
    # ])
    # deepseek.base_layer("deepseek", deepseek_base_decode, args.input_len,
    #                     args.output_len, args.batch_size, model_config, True,
    #                     False)
    # # print(deepseek_base_decode)
    # deepseek_base_decode.to_csv("./result/deepseek_base_decode_dense.csv",
    #                             index=False,
    #                             encoding="utf-8")

    # deepseek_base_decode_moe = pd.DataFrame(columns=[
    #     "Layer Name", "FLOPS", "InputA", "InputB", "Output", "FLOPS", "OP/B",
    #     "Execution_time"
    # ])
    # print(f"\n\ndeepseek_base decode moe\n")
    # deepseek.base_layer("deepseek", deepseek_base_decode_moe, args.input_len,
    #                     args.output_len, args.batch_size, model_config, True,
    #                     True)
    # # print(deepseek_base_decode_moe)
    # deepseek_base_decode_moe.to_csv("./result/deepseek_base_decode_moe.csv",
    #                                 index=False,
    #                                 encoding="utf-8")

    # df_w_uk_first = pd.DataFrame(columns=["Layer Name", "FLOPS", "InputA", "InputB", "Output"])
    # print(f"\n\ndeepseek_w_uk_first prefill dense\n")
    # deepseek.w_uk_first_layer("deepseek", df_w_uk_first, input_seq_len, output_seq_len, batch_size, model_config, False, False)
    # print(df_w_uk_first)
    # df_w_uk_first.to_csv("./result/deepseek_w_uk_first_prefill_dense.csv", index=False, encoding="utf-8")
    # df_w_uk_first = pd.DataFrame(columns=[
    #     "Layer Name", "FLOPS", "InputA", "InputB", "Output", "FLOPS", "OP/B",
    #     "Execution_time"
    # ])
    # deepseek.w_uk_first_layer("deepseek", df_w_uk_first, args.input_len,
    #                           args.output_len, args.batch_size, model_config,
    #                           False, False)
    # df_w_uk_first.to_csv("./result/absorb/deepseek_w_uk_first_prefill_dense.csv",
    #                      index=False,
    #                      encoding="utf-8")
    # create_time_graph(df_w_uk_first, "/absorb/deepseek_w_uk_first_prefill_dense")

    # df_w_uk_first_prefill_moe = pd.DataFrame(columns=[
    #     "Layer Name", "FLOPS", "InputA", "InputB", "Output", "FLOPS", "OP/B",
    #     "Execution_time"
    # ])
    # deepseek.w_uk_first_layer("deepseek", df_w_uk_first_prefill_moe,
    #                           args.input_len, args.output_len, args.batch_size,
    #                           model_config, False, True)
    # df_w_uk_first_prefill_moe.to_csv(
    #     "./result/absorb/deepseek_w_uk_first_prefill_moe.csv",
    #     index=False,
    #     encoding="utf-8")
    # create_time_graph(df_w_uk_first_prefill_moe, "/absorb/df_w_uk_first_prefill_moe")

    # df_w_uk_first_decode = pd.DataFrame(columns=[
    #     "Layer Name", "FLOPS", "InputA", "InputB", "Output", "FLOPS", "OP/B",
    #     "Execution_time"
    # ])
    # deepseek.w_uk_first_layer("deepseek", df_w_uk_first_decode, args.input_len,
    #                           args.output_len, args.batch_size, model_config,
    #                           True, False)
    # df_w_uk_first_decode.to_csv("./result/absorb/deepseek_w_uk_first_decode_dense.csv",
    #                             index=False,
    #                             encoding="utf-8")
    # create_time_graph(df_w_uk_first_decode, "/absorb/deepseek_w_uk_first_decode_dense")

    # df_w_uk_first_decode_moe = pd.DataFrame(columns=[
    #     "Layer Name", "FLOPS", "InputA", "InputB", "Output", "FLOPS", "OP/B",
    #     "Execution_time"
    # ])
    # deepseek.w_uk_first_layer("deepseek", df_w_uk_first_decode_moe,
    #                           args.input_len, args.output_len, args.batch_size,
    #                           model_config, True, True)
    # df_w_uk_first_decode_moe.to_csv(
    #     "./result/absorb/deepseek_w_uk_first_decode_moe.csv",
    #     index=False,
    #     encoding="utf-8")

    # create_time_graph(df_w_uk_first_decode_moe, "/absorb/deepseek_w_uk_first_decode_moe")


if __name__ == "__main__":
    main()

#python main.py --input_seq_len 1024 --output-len 1 --batch-size 1 --data-size 2 --model-num 0 --tensor-parallelism 1
