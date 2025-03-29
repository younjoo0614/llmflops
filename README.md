* TODO
1. FlashAttention, FlashMLA
2. MoE 
3. gate up 같이 하기

* How to use
```
python main.py --input-len {input_len} --output-len {output_len} --batch-size {batch_size} --data-size {data-size}
```

Assuming DP * TP devices, DP * TP expert parallelism 

Edit 'device_config.json' to change device specs