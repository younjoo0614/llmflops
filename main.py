import argparse
import json
import config
from config import load_layer_config, load_model_config, create_layer_dataframe
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

            
def main(input_seq_len, output_seq_len, batch_size, data_size, model_num):

    layer_config = load_layer_config()
    # print("===== Layer Configuration =====")
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
    df = create_layer_dataframe(layer_config)
    # print(df)
    # df.to_csv("temp.csv", index=False, encoding="utf-8")
    deepseek_base = Model("deepseek_base", df)
    deepseek_base.base_layer("deepseek_base", df, input_seq_len, batch_size, model_config, False, False)
    
    print(f"\n\ndeepseek_base decode dense\n")
    deepseek_base.base_layer("deepseek_base", df, input_seq_len, batch_size, model_config, True, False)

    print(f"\n\ndeepseek_base prefill moe\n")
    deepseek_base.base_layer("deepseek_base", df, input_seq_len, batch_size, model_config, False, True)

    print(f"\n\ndeepseek_base deocde moe\n")
    deepseek_base.base_layer("deepseek_base", df, input_seq_len, batch_size, model_config, True, True)


    
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




#todo - 시트 option 받는거 추가하기