from matrix import Matrix
from layer import Layer

class Model:
    def __init__(self, name, df):
        self.name = name
        self.df = df
    
    def base_layer(self, name, df, input_seq_len, batch_size, model_config):
        #input matrix 
        input_matrix = Matrix(input_seq_len, model_config["d_emb"], batch_size)

        #weidht matrix
        weight_dq = Matrix(model_config["d_emb"], model_config["q lora rank"])
        weight_dkv = Matrix(model_config["d_emb"], model_config["kv lora rank"])
        weight_uq = Matrix(model_config["q lora rank"], model_config["n_heads"]*model_config["qk nope head dim"])
        weight_uk = Matrix(model_config["kv lora rank"], model_config["n_heads"]*model_config["qk nope head dim"])
        weight_uv = Matrix(model_config["kv lora rank"], model_config["n_heads"]*model_config["qk nope head dim"])
        weight_rq = Matrix(model_config["q lora rank"], model_config["n_heads"]*model_config["qk rope head dim"])
        weight_rk = Matrix(model_config["d_emb"], model_config["qk rope head dim"])
        weight_op = Matrix(model_config["n_heads"]*model_config["qk nope head dim"], model_config["d_emb"])
        weight_gate = Matrix(model_config["d_emb"], model_config["intermediate dim"])
        weight_up = Matrix(model_config["d_emb"], model_config["intermediate dim"])
        weight_down = Matrix(model_config["intermediate dim"], model_config["d_emb"])

        #Activation matrix
        hidden_state = Matrix(1,1,1)
        compressed_q = Matrix(1,1,1)
        decompressed_q  = Matrix(1,1,1)
        compressed_kv = Matrix(1,1,1)
        decompressed_k = Matrix(1,1,1)
        decompressed_v = Matrix(1,1,1)
        ropped_k = Matrix(1,1,1)
        ropped_q = Matrix(1,1,1)
        duplicated_ropped_k = Matrix(1,1,1)
        concated_q = Matrix(1,1,1)
        concated_k = Matrix(1,1,1)
        scored_result = Matrix(1,1,1)
        mask_scale_softmax_result = Matrix(1,1,1)
        context_result = Matrix(1,1,1)
        out_proj_result = Matrix(1,1,1)
        residual_addition_result = Matrix(1,1,1)
        post_attn_norm_result = Matrix(1,1,1)
        gate_proj_result = Matrix(1,1,1)
        up_proj_result = Matrix(1,1,1)
        silu_result = Matrix(1,1,1)
        down_proj_result = Matrix(1,1,1)
        result_vector = Matrix(1,1,1)

        
        base_layers = [
            Layer("pre_attn_norm", input_matrix, None, hidden_state),
            Layer("query_down", input_matrix, weight_dq, compressed_q),
            Layer("attn_norm_1", compressed_q, None, compressed_q),
            Layer("query_up", compressed_q, weight_uq, decompressed_q),
            Layer("kv_down", hidden_state, weight_dkv, compressed_kv),
            Layer("attn_norm_2", compressed_kv, None, compressed_kv),
            Layer("k_up", compressed_kv, weight_uk, decompressed_k),
            Layer("v_up", compressed_kv, weight_uv, decompressed_v),
            Layer("k_rope_w", hidden_state, weight_rk, ropped_k),
            Layer("q_rope_w", compressed_q, weight_rq, ropped_q),
            Layer("k_rope", ropped_k, None, ropped_k),
            Layer("q_rope", ropped_q, None, ropped_q),
            Layer("score", decompressed_q.concat(ropped_q, False), decompressed_k.concat(duplicated_ropped_k, False), mask_scale_softmax_result),
            Layer("mask_scale_softmax", mask_scale_softmax_result, None, mask_scale_softmax_result),
            Layer("context_head", mask_scale_softmax_result, decompressed_v, context_result),
            Layer("out_proj", context_result, weight_op, out_proj_result),
            Layer("residual_addition", out_proj_result, None, residual_addition_result),
            Layer("post_attn_norm", residual_addition_result, None, post_attn_norm_result),
            Layer("gate_proj", post_attn_norm_result, weight_gate, gate_proj_result),
            Layer("up_proj", post_attn_norm_result, weight_up, up_proj_result),
            Layer("silu", up_proj_result, None, silu_result),
            Layer("down_proj", silu_result, weight_down, down_proj_result),
            Layer("residual_addition2", down_proj_result, None, result_vector),
        ]


        for layer in base_layers:
            print(layer.name)
            result = layer.forward()
            layer.output.update(result)
            # layer.output = Matrix(result.rows, result.cols, result.batch)
            
            print(layer.flops)

            if layer.name == "context_head":
                layer.output.cols = layer.output.cols * model_config["n_heads"]
                layer.output.batch = layer.output.batch / model_config["n_heads"]
            elif layer.name == "q_rope":
                duplicated_ropped_k = Matrix(input_seq_len, model_config["qk rope head dim"]*model_config["n_heads"])
                concated_q = decompressed_q.concat(ropped_q, False)
                concated_k = decompressed_k.concat(duplicated_ropped_k, False)

        
        print(result_vector.total_flops)
