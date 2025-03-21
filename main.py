import argparse
import json
import config
from config import load_layer_config, load_model_config, create_layer_dataframe
from matrix import Matrix
from layer import Layer
from model import Model
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

            
def main(input_seq_len, output_seq_len, batch_size, data_size, model_num):

    # print_layer_config(layer_config)
    model_config = load_model_config(model_num)
    # print("===== Model Configuration =====")
    # print(json.dumps(model_config, indent=4))

    print("===== Configuration =====")
    print(f"Input Sequence Length: {input_seq_len}")
    print(f"Output Sequence Length: {output_seq_len}")
    print(f"Batch Size: {batch_size}")
    print(f"Data Size: {data_size}")
    print(f"Model Num: {model_num}")
    print(f"Model Name: {model_config['Model Name']}\n\n")
    print(f"deepseek_base prefiil dense\n")
    
    deepseek = Model("deepseek")

    deepseek_base = pd.DataFrame(columns=["Layer Name", "FLOPS", "InputA", "InputB", "Output"])
    print(f"\n\ndeepseek_w_uk_first prefill dense\n")
    deepseek.base_layer("deepseek", deepseek_base, input_seq_len, output_seq_len, batch_size, model_config, False, False)
    print(deepseek_base)
    deepseek_base.to_csv("./result/deepseek_base_prefill_dense.csv", index=False, encoding="utf-8")

    deepseek_base_prefill_moe = pd.DataFrame(columns=["Layer Name", "FLOPS", "InputA", "InputB", "Output"])
    print(f"\n\ndeepseek_w_uk_first prefill moe\n")
    deepseek.base_layer("deepseek", deepseek_base_prefill_moe, input_seq_len, output_seq_len, batch_size, model_config, False, True)
    print(deepseek_base_prefill_moe)
    deepseek_base_prefill_moe.to_csv("./result/deepseek_base_prefill_moe.csv", index=False, encoding="utf-8")

    deepseek_base_decode = pd.DataFrame(columns=["Layer Name", "FLOPS", "InputA", "InputB", "Output"])
    print(f"\n\ndeepseek_w_uk_first decode dense\n")
    deepseek.base_layer("deepseek", deepseek_base_decode, input_seq_len, output_seq_len, batch_size, model_config, True, False)
    print(deepseek_base_decode)
    deepseek_base_decode.to_csv("./result/deepseek_base_decode_dense.csv", index=False, encoding="utf-8")

    deepseek_base_decode_moe = pd.DataFrame(columns=["Layer Name", "FLOPS", "InputA", "InputB", "Output"])
    print(f"\n\ndeepseek_w_uk_first decode moe\n")
    deepseek.base_layer("deepseek", deepseek_base_decode_moe, input_seq_len, output_seq_len, batch_size, model_config, True, True)
    print(deepseek_base_decode_moe)
    deepseek_base_decode_moe.to_csv("./result/deepseek_base_decode_moe.csv", index=False, encoding="utf-8")





    df_w_uk_first = pd.DataFrame(columns=["Layer Name", "FLOPS", "InputA", "InputB", "Output"])
    print(f"\n\ndeepseek_w_uk_first prefill dense\n")
    deepseek.w_uk_first_layer("deepseek", df_w_uk_first, input_seq_len, output_seq_len, batch_size, model_config, False, False)
    print(df_w_uk_first)
    df_w_uk_first.to_csv("./result/deepseek_w_uk_first_prefill_dense.csv", index=False, encoding="utf-8")

    df_w_uk_first_prefill_moe = pd.DataFrame(columns=["Layer Name", "FLOPS", "InputA", "InputB", "Output"])
    print(f"\n\ndeepseek_w_uk_first prefill moe\n")
    deepseek.w_uk_first_layer("deepseek", df_w_uk_first_prefill_moe, input_seq_len, output_seq_len, batch_size, model_config, False, True)
    print(df_w_uk_first_prefill_moe)
    df_w_uk_first_prefill_moe.to_csv("./result/deepseek_w_uk_first_prefill_moe.csv", index=False, encoding="utf-8")

    df_w_uk_first_decode = pd.DataFrame(columns=["Layer Name", "FLOPS", "InputA", "InputB", "Output"])
    print(f"\n\ndeepseek_w_uk_first decode dense\n")
    deepseek.w_uk_first_layer("deepseek", df_w_uk_first_decode, input_seq_len, output_seq_len, batch_size, model_config, True, False)
    print(df_w_uk_first_decode)
    df_w_uk_first_decode.to_csv("./result/deepseek_w_uk_first_decode_dense.csv", index=False, encoding="utf-8")

    df_w_uk_first_decode_moe = pd.DataFrame(columns=["Layer Name", "FLOPS", "InputA", "InputB", "Output"])
    print(f"\n\ndeepseek_w_uk_first decode moe\n")
    deepseek.w_uk_first_layer("deepseek", df_w_uk_first_decode_moe, input_seq_len, output_seq_len, batch_size, model_config, True, True)
    print(df_w_uk_first_decode_moe)
    df_w_uk_first_decode_moe.to_csv("./result/deepseek_w_uk_first_decode_moe.csv", index=False, encoding="utf-8")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process model training with configurations")
    parser.add_argument("--input_seq_len", type=int, required=True, help="Input Sequence Length")
    parser.add_argument("--output_seq_len", type=int, required=True, help="Output Sequence Length")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch Size")
    parser.add_argument("--data_size", type=int, required=True, help="Data Size")
    parser.add_argument("--model_num", type=int, required=True, help="Model Num")

    args = parser.parse_args()
    main(args.input_seq_len, args.output_seq_len, args.batch_size, args.data_size, args.model_num)

#python main.py --input_seq_len 1024 --output_seq_len 1 --batch_size 1 --data_size 2 --model_num 0
