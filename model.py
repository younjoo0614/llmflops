import json
import pandas as pd

from matrix import Matrix
from layer import Layer
from tp_type import TPType

class Model:

    def __init__(self, name):
        self.name = name

    def base_layer(self, name, df, input_len, output_len, batch_size, tp_degree,
                   model_config, decode_flag, moe_flag):

        #input matrix
        if decode_flag == False:  #prefill
            input_matrix = Matrix(input_len, model_config["d_emb"], batch_size)
        else:  #decode
            input_matrix = Matrix(1, model_config["d_emb"], batch_size)
            # input_matrix = Matrix((output_len+1)/2, model_config["d_emb"], batch_size)

        #weight matrix
        weight_dq = Matrix(model_config["d_emb"], model_config["q lora rank"])
        weight_dkv = Matrix(model_config["d_emb"], model_config["kv lora rank"])
        weight_uq = Matrix(model_config["q lora rank"], model_config["n_heads"] * model_config["qk nope head dim"])
        weight_uk = Matrix(model_config["kv lora rank"], model_config["n_heads"] * model_config["qk nope head dim"])
        weight_uv = Matrix(model_config["kv lora rank"], model_config["n_heads"] * model_config["qk nope head dim"])
        weight_rq = Matrix(model_config["q lora rank"], model_config["n_heads"] * model_config["qk rope head dim"])
        weight_rk = Matrix(model_config["d_emb"], model_config["qk rope head dim"])
        weight_op = Matrix(model_config["n_heads"] * model_config["qk nope head dim"], model_config["d_emb"])
        weight_gate = Matrix(model_config["d_emb"], model_config["intermediate dim"])
        weight_up = Matrix(model_config["d_emb"], model_config["intermediate dim"])
        weight_down = Matrix(model_config["intermediate dim"], model_config["d_emb"])
        weight_router = Matrix(model_config['d_emb'], model_config['n_experts'])
        weight_gate_routed = Matrix(model_config['d_emb'], model_config['moe intermediate dim'])
        weight_up_routed = Matrix(model_config['d_emb'], model_config['moe intermediate dim'])
        weight_down_routed = Matrix(model_config['moe intermediate dim'], model_config['d_emb'])
        weight_gate_shared = Matrix(model_config['d_emb'], model_config['moe intermediate dim'])
        weight_up_shared = Matrix(model_config['d_emb'], model_config['moe intermediate dim'])
        weight_down_shared = Matrix(model_config['moe intermediate dim'],model_config['d_emb'])

        #Activation matrix
        if decode_flag:
            seq_len = 1
        else:
            seq_len = input_len
        
        hidden_state = Matrix(seq_len, model_config["d_emb"], batch_size)
        compressed_q = Matrix(seq_len, model_config["q lora rank"], batch_size)
        decompressed_q = Matrix(seq_len, model_config["n_heads"] * model_config["qk nope head dim"], batch_size)
        compressed_kv = Matrix(seq_len, model_config["kv lora rank"], batch_size)
        decompressed_k = Matrix(seq_len, model_config["n_heads"] * model_config["qk nope head dim"], batch_size)
        decompressed_v = Matrix(seq_len, model_config["n_heads"] * model_config["qk nope head dim"], batch_size)
        ropped_k = Matrix(seq_len, model_config["qk rope head dim"], batch_size)
        ropped_q = Matrix(seq_len, model_config["n_heads"] * model_config["qk rope head dim"], batch_size)
        duplicated_ropped_k = Matrix(seq_len, model_config["n_heads"]*model_config["qk rope head dim"], batch_size)
        concated_q = Matrix(seq_len, model_config["n_heads"] * model_config["qk nope head dim"] + model_config["n_heads"]*model_config["qk rope head dim"], batch_size)
        if decode_flag:
            concated_k = Matrix((output_len + 1) / 2 + input_len, model_config["n_heads"] * model_config["qk nope head dim"] + model_config["n_heads"]*model_config["qk rope head dim"], batch_size)
            scored_result = Matrix(seq_len, (output_len + 1) / 2 + input_len, batch_size)
            mask_scale_softmax_result = Matrix(seq_len, (output_len + 1) / 2 + input_len, batch_size)
            context_result = Matrix(seq_len, model_config["qk nope head dim"], batch_size)
        else:
            concated_k = Matrix(seq_len, model_config["n_heads"] * model_config["qk nope head dim"] + model_config["n_heads"]*model_config["qk rope head dim"], batch_size)
            scored_result = Matrix(seq_len, seq_len, batch_size)
            mask_scale_softmax_result = Matrix(seq_len, seq_len, batch_size)
            context_result = Matrix(seq_len, model_config["qk nope head dim"], batch_size)
        out_proj_result = Matrix(seq_len, model_config["d_emb"], batch_size)
        residual_addition_result = Matrix(seq_len, model_config["d_emb"], batch_size)
        post_attn_norm_result = Matrix(seq_len, model_config["d_emb"], batch_size)
        gate_proj_result = Matrix(seq_len, model_config["intermediate dim"], batch_size)
        up_proj_result = Matrix(seq_len, model_config["intermediate dim"], batch_size)
        silu_result = Matrix(seq_len, model_config["intermediate dim"], batch_size)
        down_proj_result = Matrix(seq_len, model_config['d_emb'], batch_size)
        result_vector = Matrix(seq_len, model_config['d_emb'], batch_size)
        routed_result = Matrix(seq_len, model_config['n_experts'], batch_size)
        gate_routed_result = Matrix(seq_len*model_config['top-k']/model_config['n_experts'], model_config['moe intermediate dim'], batch_size)
        up_routed_result = Matrix(seq_len*model_config['top-k']/model_config['n_experts'], model_config['moe intermediate dim'], batch_size)
        silu_routed_result = Matrix(seq_len*model_config['top-k']/model_config['n_experts'], model_config['moe intermediate dim'], batch_size)
        down_routed_result = Matrix(seq_len*model_config['top-k']/model_config['n_experts'], model_config['d_emb'], batch_size)
        gate_shared_result = Matrix(seq_len, model_config['moe intermediate dim'], batch_size)
        up_shared_result = Matrix(seq_len, model_config['moe intermediate dim'], batch_size)
        silu_shared_result = Matrix(seq_len, model_config['moe intermediate dim'], batch_size)
        down_shared_result = Matrix(seq_len, model_config['d_emb'], batch_size)

        base_layers = [
            Layer("pre_attn_norm", input_matrix, None, hidden_state),
            Layer("query_down", input_matrix, weight_dq, compressed_q, None, tp_degree, False),
            Layer("attn_norm_1", compressed_q, None, compressed_q, None, tp_degree, False), 
            Layer("query_up", compressed_q, weight_uq, decompressed_q, TPType.COL, tp_degree),
            Layer("kv_down", hidden_state, weight_dkv, compressed_kv, None, tp_degree, False),
            Layer("attn_norm_2", compressed_kv, None, compressed_kv, None, tp_degree, False), #확인
            Layer("k_up", compressed_kv, weight_uk, decompressed_k, TPType.COL, tp_degree, True),
            Layer("v_up", compressed_kv, weight_uv, decompressed_v, TPType.COL, tp_degree, True),
            Layer("k_rope_w", hidden_state, weight_rk, ropped_k, None, tp_degree),
            Layer("q_rope_w", compressed_q, weight_rq, ropped_q, TPType.COL, tp_degree),
            Layer("k_rope", ropped_k, None, ropped_k, None, tp_degree),
            Layer("q_rope", ropped_q, None, ropped_q, TPType.COL, tp_degree),
            Layer("score", concated_q, concated_k, scored_result, TPType.HEAD_COL_COL, tp_degree),
            Layer("mask_scale_softmax", scored_result, None, mask_scale_softmax_result, TPType.NONE, tp_degree),
            Layer("context_head", mask_scale_softmax_result, decompressed_v, context_result, TPType.COL, tp_degree),
            Layer("out_proj", context_result, weight_op, out_proj_result, TPType.ROW, tp_degree),
            Layer("residual_addition", out_proj_result, None, residual_addition_result, None, tp_degree),
            Layer("post_attn_norm", residual_addition_result, None, post_attn_norm_result, None, tp_degree)
        ]
        base_non_moe_ffn_layers = [
            Layer("gate_proj", post_attn_norm_result, weight_gate, gate_proj_result, TPType.COL, tp_degree),
            Layer("up_proj", post_attn_norm_result, weight_up, up_proj_result, TPType.COL, tp_degree),
            Layer("silu", up_proj_result, None, silu_result, TPType.COL, tp_degree),
            Layer("down_proj", silu_result, weight_down, down_proj_result, TPType.ROW, tp_degree),
            Layer("residual_addition2", down_proj_result, None, result_vector, None, tp_degree)
        ]

        base_moe_ffn_layers = [
            Layer("gate_shared", post_attn_norm_result, weight_gate_shared, gate_shared_result),
            Layer("up_shared", post_attn_norm_result, weight_up_shared, up_shared_result),
            Layer("silu_shared", up_shared_result, None, silu_shared_result),
            Layer("down_shared", silu_shared_result, weight_down_shared, down_shared_result),
            Layer("router", post_attn_norm_result, weight_router, routed_result),
            Layer("gate_routed", post_attn_norm_result, weight_gate_routed, gate_routed_result),
            Layer("up_routed", post_attn_norm_result, weight_up_routed, up_routed_result),
            Layer("silu_routed", up_routed_result, None, silu_routed_result),
            Layer("down_routed", silu_routed_result, weight_down_routed, down_routed_result),
            Layer("residual_addition2", down_shared_result, None, result_vector)
        ]

        if moe_flag == False:
            base_layers = base_layers + base_non_moe_ffn_layers
        else:
            base_layers = base_layers + base_moe_ffn_layers

        for layer in base_layers:
            if layer.name == "score" and decode_flag == True:
                layer.inputB.rows = (output_len + 1) / 2 + input_len
            elif layer.name == "context_head" and decode_flag == True:
                layer.inputB.rows = (output_len + 1) / 2 + input_len
            # elif layer.name == "gate_routed":
            #     layer.inputA.rows = layer.inputA.rows * model_config['top-k'] * layer.inputA.batch / model_config['n_experts']
            #     layer.inputA.batch = 1
            #     if layer.inputA.rows < 1:
            #         layer.inputA.rows = 1
            elif layer.name == "gate_routed":
                if decode_flag == False:
                    layer.inputA.rows = layer.inputA.rows * model_config['top-k'] / model_config['n_experts']
                else:
                    layer.inputA.rows = layer.inputA.rows * model_config['top-k'] * layer.inputA.batch / model_config['n_experts']
                    layer.inputA.batch = 1
                    if layer.inputA.rows < 1:
                        layer.inputA.rows = 1

            print(layer.name)
            print(layer.inputA)
            print(layer.inputB)
            result = layer.forward()
            layer.output.reshape(result)

            if layer.name == "q_rope":
                if decode_flag == False:
                    duplicated_ropped_k = Matrix(input_len, model_config["qk rope head dim"] * model_config["n_heads"])
                    concated_q = decompressed_q.concat(ropped_q, False)
                    concated_k = decompressed_k.concat(duplicated_ropped_k, False)

                else:
                    # duplicated_ropped_k = Matrix((output_len+1)/2, model_config["qk rope head dim"]*model_config["n_heads"])
                    duplicated_ropped_k = Matrix(1, model_config["qk rope head dim"] *  model_config["n_heads"])
                    concated_q = decompressed_q.concat(ropped_q, False)
                    concated_k = decompressed_k.concat(duplicated_ropped_k, False)

            
            df.loc[len(df)] = [
                layer.name,
                layer.get_flops(),
                layer.inputA.get_size(),
                layer.inputB.get_size() if layer.inputB is not None else "",
                layer.output.get_size(),
                layer.get_flops(),
                layer.get_op_per_byte(),
                layer.get_execution_time()
            ]

            # if decode_flag == False:
            #     df.loc[len(df)] = [
            #         layer.name,
            #         layer.get_flops(),
            #         layer.inputA.get_size(),
            #         layer.inputB.get_size() if layer.inputB is not None else "",
            #         layer.output.get_size(),
            #         layer.get_flops(),
            #         layer.get_op_per_byte(),
            #         layer.get_execution_time()
            #     ]
            # else:
            #     df.loc[len(df)] = [
            #         layer.name,
            #         layer.get_flops(),
            #         layer.inputA.get_size(),
            #         layer.inputB.get_size() if layer.inputB is not None else "",
            #         layer.output.get_size(),
            #         layer.get_flops(),
            #         layer.get_op_per_byte(),
            #         layer.get_execution_time()
            #     ]
            # layer.reset_parallelism()

        total_flops = df["FLOPS"].sum()
        df.loc[len(df)] = ["Total FLOPS", total_flops, "", "", "", "", "", ""]
        df["Execution_time"] = pd.to_numeric(df["Execution_time"], errors="coerce")
        total_time = df["Execution_time"].sum()
        df["Execution_time(%)"] = (df["Execution_time"] / total_time) * 100
        df["Execution_time(%)"] = df["Execution_time(%)"].round(2)
        Matrix.reset_flops()

    def w_uk_first_layer(self, name, df, input_len, output_len, batch_size, tp_degree,
                         model_config, decode_flag, moe_flag):

        #input matrix
        if decode_flag == False:  #prefill
            input_matrix = Matrix(input_len, model_config["d_emb"], batch_size)
        else:  #decode
            input_matrix = Matrix(1, model_config["d_emb"], batch_size)

        #weight matrix
        weight_dq = Matrix(model_config["d_emb"], model_config["q lora rank"])
        weight_dkv = Matrix(model_config["d_emb"], model_config["kv lora rank"])
        weight_uq = Matrix(
            model_config["q lora rank"],
            model_config["n_heads"] * model_config["qk nope head dim"])
        weight_uk = Matrix(model_config["kv lora rank"], model_config["n_heads"] * model_config["qk nope head dim"])
        weight_uv = Matrix(model_config["kv lora rank"], model_config["n_heads"] * model_config["qk nope head dim"])
        weight_rq = Matrix(model_config["q lora rank"], model_config["n_heads"] * model_config["qk rope head dim"])
        weight_rk = Matrix(model_config["d_emb"], model_config["qk rope head dim"])
        weight_op = Matrix(model_config["n_heads"] * model_config["qk nope head dim"], model_config["d_emb"])
        weight_gate = Matrix(model_config["d_emb"], model_config["intermediate dim"])
        weight_up = Matrix(model_config["d_emb"], model_config["intermediate dim"])
        weight_down = Matrix(model_config["intermediate dim"], model_config["d_emb"])
        weight_router = Matrix(model_config['d_emb'], model_config['n_experts'])
        weight_gate_routed = Matrix(model_config['d_emb'], model_config['moe intermediate dim'])
        weight_up_routed = Matrix(model_config['d_emb'], model_config['moe intermediate dim'])
        weight_down_routed = Matrix(model_config['moe intermediate dim'], model_config['d_emb'])
        weight_gate_shared = Matrix(model_config['d_emb'], model_config['moe intermediate dim'])
        weight_up_shared = Matrix(model_config['d_emb'], model_config['moe intermediate dim'])
        weight_down_shared = Matrix(model_config['moe intermediate dim'], model_config['d_emb'])

        #Activation matrix
        if decode_flag:
            seq_len = 1
        else:
            seq_len = input_len
        
        hidden_state = Matrix(seq_len, model_config["d_emb"], batch_size)
        compressed_q = Matrix(seq_len, model_config["q lora rank"], batch_size)
        decompressed_q = Matrix(seq_len, model_config["n_heads"] * model_config["qk nope head dim"], batch_size)
        compressed_kv = Matrix(seq_len, model_config["kv lora rank"], batch_size)
        ropped_k = Matrix(seq_len, model_config["qk rope head dim"], batch_size)
        ropped_q = Matrix(seq_len, model_config["n_heads"] * model_config["qk rope head dim"], batch_size)
        mask_scale_softmax_result = Matrix(seq_len, seq_len, batch_size)

        context_result = Matrix(seq_len*model_config['n_heads'], model_config["kv lora rank"], batch_size)

        out_proj_result = Matrix(seq_len, model_config["n_heads"] * model_config["qk nope head dim"], batch_size)
        residual_addition_result = Matrix(seq_len, model_config["n_heads"] * model_config["qk nope head dim"], batch_size)
        post_attn_norm_result = Matrix(seq_len, model_config["n_heads"] * model_config["qk nope head dim"], batch_size)
        gate_proj_result = Matrix(seq_len, model_config["intermediate dim"], batch_size)
        up_proj_result = Matrix(seq_len, model_config["intermediate dim"], batch_size)
        silu_result = Matrix(seq_len, model_config["intermediate dim"], batch_size)
        down_proj_result = Matrix(seq_len, model_config['d_emb'], batch_size)
        result_vector = Matrix(seq_len, model_config['d_emb'], batch_size)
        routed_result = Matrix(seq_len, model_config['n_experts'], batch_size)
        gate_routed_result = Matrix(seq_len*model_config['top-k']/model_config['n_experts'], model_config['moe intermediate dim'], batch_size)
        up_routed_result = Matrix(seq_len*model_config['top-k']/model_config['n_experts'], model_config['moe intermediate dim'], batch_size)
        silu_routed_result = Matrix(seq_len*model_config['top-k']/model_config['n_experts'], model_config['moe intermediate dim'], batch_size)
        down_routed_result = Matrix(seq_len*model_config['top-k']/model_config['n_experts'], model_config['d_emb'], batch_size)
        gate_shared_result = Matrix(seq_len, model_config['moe intermediate dim'], batch_size)
        up_shared_result = Matrix(seq_len, model_config['moe intermediate dim'], batch_size)
        silu_shared_result = Matrix(seq_len, model_config['moe intermediate dim'], batch_size)
        down_shared_result = Matrix(seq_len, model_config['d_emb'], batch_size)
        transposed_k_up_result = Matrix(seq_len, model_config["kv lora rank"], batch_size*model_config['n_heads'])
        if decode_flag:
            score_NOPE_result = Matrix(seq_len*model_config['n_heads'], (output_len + 1) / 2 + input_len ,batch_size)
            score_ROPE_result = Matrix(seq_len*model_config['n_heads'], (output_len + 1) / 2, batch_size)
            v_up_context_result = Matrix(seq_len,model_config["qk nope head dim"], batch_size)
            out_proj_context_result = Matrix(seq_len, model_config['d_emb'], batch_size)

        else:
            score_NOPE_result = Matrix(seq_len*model_config['n_heads'], seq_len ,batch_size)
            score_ROPE_result = Matrix(seq_len*model_config['n_heads'], seq_len, batch_size)
            v_up_context_result = Matrix(seq_len,model_config["qk nope head dim"], batch_size )
            out_proj_context_result = Matrix(seq_len, model_config['d_emb'], batch_size )

        w_uk_first_layers = [
            Layer("pre_attn_norm", input_matrix, None, hidden_state, None, tp_degree),
            Layer("query_down", input_matrix, weight_dq, compressed_q, None, tp_degree),
            Layer("k_rope_w", hidden_state, weight_rk, ropped_k, None, tp_degree),
            Layer("kv_down", hidden_state, weight_dkv, compressed_kv, None, tp_degree),
            Layer("norm_for_compressed_q", compressed_q, None, compressed_q, None, tp_degree),
            Layer("norm_for_compressed_kv", compressed_kv, None, compressed_kv, None, tp_degree),
            Layer("q_rope_w", compressed_q, weight_rq, ropped_q, TPType.COL, tp_degree),
            Layer("query_up", compressed_q, weight_uq, decompressed_q, TPType.COL, tp_degree),
            Layer("transposed (k up proj)", decompressed_q, weight_uk, transposed_k_up_result, TPType.HEAD_COL_COL, tp_degree),
            Layer("q_rope", ropped_q, None, ropped_q, TPType.COL, tp_degree),
            Layer("k_rope", ropped_k, None, ropped_k, None, tp_degree),
            Layer("score layer for RoPE", ropped_q, ropped_k, score_ROPE_result, TPType.ROW_IN, tp_degree),
            Layer("score layer for NoPE", transposed_k_up_result, compressed_kv,score_NOPE_result, TPType.ROW_IN, tp_degree),
            Layer("mask_scale_softmax", score_ROPE_result, None,mask_scale_softmax_result, TPType.ROW_IN, tp_degree),
            Layer("context_matmul", mask_scale_softmax_result, compressed_kv, context_result, TPType.ROW_IN, tp_degree),
            Layer("v_up_proj_context", context_result, weight_uv, v_up_context_result, TPType.HEAD_ROW_COL, tp_degree),
            Layer("out_proj", v_up_context_result, weight_op, out_proj_context_result, TPType.ROW, tp_degree),
            Layer("residual_addition", out_proj_context_result, None, residual_addition_result, None, tp_degree),
            Layer("post_attn_norm", residual_addition_result, None, post_attn_norm_result, tp_degree)
        ]

        w_uk_first_non_moe_ffn_layers = [
            Layer("gate_proj", post_attn_norm_result, weight_gate, gate_proj_result, TPType.COL, tp_degree),
            Layer("up_proj", post_attn_norm_result, weight_up, up_proj_result, TPType.COL, tp_degree),
            Layer("silu", up_proj_result, None, silu_result, TPType.COL, tp_degree),
            Layer("down_proj", silu_result, weight_down, down_proj_result, TPType.ROW, tp_degree),
            Layer("residual_addition2", down_proj_result, None, result_vector)
        ]

        w_uk_first_moe_ffn_layers = [
            Layer("gate_shared", post_attn_norm_result, weight_gate_shared, gate_shared_result),
            Layer("up_shared", post_attn_norm_result, weight_up_shared, up_shared_result),
            Layer("silu_shared", up_shared_result, None, silu_shared_result),
            Layer("down_shared", silu_shared_result, weight_down_shared, down_shared_result),
            Layer("router", post_attn_norm_result, weight_router, routed_result),
            Layer("gate_routed", post_attn_norm_result, weight_gate_routed, gate_routed_result),
            Layer("up_routed", post_attn_norm_result, weight_up_routed, up_routed_result),
            Layer("silu_routed", up_routed_result, None, silu_routed_result),
            Layer("down_routed", silu_routed_result, weight_down_routed, down_routed_result),
            Layer("residual_addition2", down_shared_result, None, result_vector)
        ]

        if moe_flag == False:
            w_uk_first_layers = w_uk_first_layers + w_uk_first_non_moe_ffn_layers
        else:
            w_uk_first_layers = w_uk_first_layers + w_uk_first_moe_ffn_layers

        for layer in w_uk_first_layers:

            if layer.name == "score layer for NoPE":
                layer.inputB.transpose()
                if decode_flag == True:
                    # layer.inputB.cols = layer.inputB.cols + input_len + i
                    layer.inputB.cols = (output_len + 1) / 2 + input_len
            elif layer.name == "context_matmul":
                layer.inputB.transpose()
            elif layer.name == "v_up_proj_context":
                layer.inputA.reshape(
                    Matrix(layer.inputA.rows / model_config['n_heads'],
                           layer.inputA.cols,
                           layer.inputA.batch * model_config['n_heads']))
            elif layer.name == "gate_routed":
                if decode_flag == False:
                    layer.inputA.rows = layer.inputA.rows * model_config['top-k'] / model_config['n_experts']
                else:
                    layer.inputA.rows = layer.inputA.rows * model_config['top-k'] * layer.inputA.batch / model_config['n_experts']
                    layer.inputA.batch = 1
                    if layer.inputA.rows < 1:
                        layer.inputA.rows = 1
            elif layer.name == "score layer for RoPE" and decode_flag == True:
                # layer.inputB.cols = layer.inputB.cols + input_len  + i
                layer.inputB.cols = (output_len + 1) / 2 + input_len
            
            result = layer.forward()
            layer.output.reshape(result)
            # layer.reset_parallelism()
            # print(layer.output)

            if layer.name == "score layer for RoPE":
                layer.output.rows = input_len * model_config["n_heads"]
            #reshape after q_rope
            if layer.name == "q_rope" and decode_flag:
                layer.output.rows = 128
                layer.output.cols = 64
                ropped_k.transpose()
            elif layer.name == "q_rope" and not decode_flag:
                layer.output.rows = input_len * model_config["n_heads"]
                layer.output.cols = model_config["qk rope head dim"]
                ropped_k.transpose()
            if layer.name == "score layer for RoPE":
                if decode_flag == False:
                    layer.output.rows = input_len * model_config["n_heads"]
                else:
                    layer.output.rows = model_config["n_heads"]
            if layer.name == "out_proj_context":
                layer.output.batch = layer.output.batch / model_config["n_heads"]

            if decode_flag == False:
                df.loc[len(df)] = [
                    layer.name,
                    layer.get_flops(),
                    layer.inputA.get_size(),
                    layer.inputB.get_size() if layer.inputB is not None else "",
                    layer.output.get_size(),
                    layer.get_flops(),
                    layer.get_op_per_byte(),
                    layer.get_execution_time()
                ]
            else:
                df.loc[len(df)] = [
                    layer.name,
                    layer.get_flops(),
                    layer.inputA.get_size(),
                    layer.inputB.get_size() if layer.inputB is not None else "",
                    layer.output.get_size(),
                    layer.get_flops(),
                    layer.get_op_per_byte(),
                    layer.get_execution_time()
                ]

        total_flops = df["FLOPS"].sum()
        df.loc[len(df)] = ["Total FLOPS", total_flops, "", "", "", "", "", ""]
        df["Execution_time"] = pd.to_numeric(df["Execution_time"], errors="coerce")
        total_time = df["Execution_time"].sum()
        df["Execution_time(%)"] = (df["Execution_time"] / total_time) * 100
        df["Execution_time(%)"] = df["Execution_time(%)"].round(2)

        Matrix.reset_flops()
