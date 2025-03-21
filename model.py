from matrix import Matrix
from layer import Layer
import json


class Model:

    def __init__(self, name):
        self.name = name

    def base_layer(self, name, df, input_len, output_len, batch_size,
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
        weight_uq = Matrix(
            model_config["q lora rank"],
            model_config["n_heads"] * model_config["qk nope head dim"])
        weight_uk = Matrix(
            model_config["kv lora rank"],
            model_config["n_heads"] * model_config["qk nope head dim"])
        weight_uv = Matrix(
            model_config["kv lora rank"],
            model_config["n_heads"] * model_config["qk nope head dim"])
        weight_rq = Matrix(
            model_config["q lora rank"],
            model_config["n_heads"] * model_config["qk rope head dim"])
        weight_rk = Matrix(model_config["d_emb"],
                           model_config["qk rope head dim"])
        weight_op = Matrix(
            model_config["n_heads"] * model_config["qk nope head dim"],
            model_config["d_emb"])
        weight_gate = Matrix(model_config["d_emb"],
                             model_config["intermediate dim"])
        weight_up = Matrix(model_config["d_emb"],
                           model_config["intermediate dim"])
        weight_down = Matrix(model_config["intermediate dim"],
                             model_config["d_emb"])
        weight_router = Matrix(model_config['d_emb'], model_config['n_experts'])
        weight_gate_routed = Matrix(model_config['d_emb'],
                                    model_config['moe intermediate dim'])
        weight_up_routed = Matrix(model_config['d_emb'],
                                  model_config['moe intermediate dim'])
        weight_down_routed = Matrix(model_config['moe intermediate dim'],
                                    model_config['d_emb'])
        weight_gate_shared = Matrix(model_config['d_emb'],
                                    model_config['moe intermediate dim'])
        weight_up_shared = Matrix(model_config['d_emb'],
                                  model_config['moe intermediate dim'])
        weight_down_shared = Matrix(model_config['moe intermediate dim'],
                                    model_config['d_emb'])

        #Activation matrix
        hidden_state = Matrix(1, 1, 1)
        compressed_q = Matrix(1, 1, 1)
        decompressed_q = Matrix(1, 1, 1)
        compressed_kv = Matrix(1, 1, 1)
        decompressed_k = Matrix(1, 1, 1)
        decompressed_v = Matrix(1, 1, 1)
        ropped_k = Matrix(1, 1, 1)
        ropped_q = Matrix(1, 1, 1)
        duplicated_ropped_k = Matrix(1, 1, 1)
        concated_q = Matrix(1, 1, 1)
        concated_k = Matrix(1, 1, 1)
        scored_result = Matrix(1, 1, 1)
        mask_scale_softmax_result = Matrix(1, 1, 1)
        context_result = Matrix(1, 1, 1)
        out_proj_result = Matrix(1, 1, 1)
        residual_addition_result = Matrix(1, 1, 1)
        post_attn_norm_result = Matrix(1, 1, 1)
        gate_proj_result = Matrix(1, 1, 1)
        up_proj_result = Matrix(1, 1, 1)
        silu_result = Matrix(1, 1, 1)
        down_proj_result = Matrix(1, 1, 1)
        result_vector = Matrix(1, 1, 1)
        routed_result = Matrix(1, 1, 1)
        gate_routed_result = Matrix(1, 1, 1)
        up_routed_result = Matrix(1, 1, 1)
        silu_routed_result = Matrix(1, 1, 1)
        down_routed_result = Matrix(1, 1, 1)
        gate_shared_result = Matrix(1, 1, 1)
        up_shared_result = Matrix(1, 1, 1)
        silu_shared_result = Matrix(1, 1, 1)
        down_shared_result = Matrix(1, 1, 1)

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
            Layer("score", decompressed_q.concat(ropped_q, False),
                  decompressed_k.concat(duplicated_ropped_k, False),
                  mask_scale_softmax_result),
            Layer("mask_scale_softmax", mask_scale_softmax_result, None,
                  mask_scale_softmax_result),
            Layer("context_head", mask_scale_softmax_result, decompressed_v,
                  context_result),
            Layer("out_proj", context_result, weight_op, out_proj_result),
            Layer("residual_addition", out_proj_result, None,
                  residual_addition_result),
            Layer("post_attn_norm", residual_addition_result, None,
                  post_attn_norm_result)
        ]

        base_non_moe_ffn_layers = [
            Layer("gate_proj", post_attn_norm_result, weight_gate,
                  gate_proj_result),
            Layer("up_proj", post_attn_norm_result, weight_up, up_proj_result),
            Layer("silu", up_proj_result, None, silu_result),
            Layer("down_proj", silu_result, weight_down, down_proj_result),
            Layer("residual_addition2", down_proj_result, None, result_vector)
        ]

        base_moe_ffn_layers = [
            Layer("gate_shared", post_attn_norm_result, weight_gate_shared,
                  gate_shared_result),
            Layer("up_shared", post_attn_norm_result, weight_up_shared,
                  up_shared_result),
            Layer("silu_shared", up_shared_result, None, silu_shared_result),
            Layer("down_shared", silu_shared_result, weight_down_shared,
                  down_shared_result),
            Layer("router", post_attn_norm_result, weight_router,
                  routed_result),
            Layer("gate_routed", post_attn_norm_result, weight_gate_routed,
                  gate_routed_result),
            Layer("up_routed", post_attn_norm_result, weight_up_routed,
                  up_routed_result),
            Layer("silu_routed", up_routed_result, None, silu_routed_result),
            Layer("down_routed", silu_routed_result, weight_down_routed,
                  down_routed_result),
            Layer("residual_addition2", down_shared_result, None, result_vector)
        ]

        if moe_flag == False:
            base_layers = base_layers + base_non_moe_ffn_layers
        else:
            base_layers = base_layers + base_moe_ffn_layers

        for layer in base_layers:
            if layer.name == "score" and decode_flag == True:
                layer.inputB.rows = (output_len + 1) / 2 + input_len
            elif layer.name == "gate_routed":
                layer.inputA.rows = layer.inputA.rows * model_config[
                    'top-k'] / model_config['n_experts']
                if layer.inputA.rows < 1:
                    layer.inputA.rows = 1

            # print(layer.name, layer.flops)
            # print(layer.inputA)
            # print(layer.inputB)
            result = layer.forward()
            layer.output.reshape(result)

            if layer.name == "context_head":
                layer.output.cols = layer.output.cols * model_config["n_heads"]
                layer.output.batch = layer.output.batch / model_config["n_heads"]
            elif layer.name == "q_rope":
                if decode_flag == False:
                    duplicated_ropped_k = Matrix(
                        input_len, model_config["qk rope head dim"] *
                        model_config["n_heads"])
                    concated_q = decompressed_q.concat(ropped_q, False)
                    concated_k = decompressed_k.concat(duplicated_ropped_k,
                                                       False)
                else:
                    # duplicated_ropped_k = Matrix((output_len+1)/2, model_config["qk rope head dim"]*model_config["n_heads"])
                    duplicated_ropped_k = Matrix(
                        1, model_config["qk rope head dim"] *
                        model_config["n_heads"])
                    concated_q = decompressed_q.concat(ropped_q, False)
                    concated_k = decompressed_k.concat(duplicated_ropped_k,
                                                       False)

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
        Matrix.reset_flops()

    def w_uk_first_layer(self, name, df, input_len, output_len, batch_size,
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
        weight_uk = Matrix(
            model_config["kv lora rank"],
            model_config["n_heads"] * model_config["qk nope head dim"])
        weight_uv = Matrix(
            model_config["kv lora rank"],
            model_config["n_heads"] * model_config["qk nope head dim"])
        weight_rq = Matrix(
            model_config["q lora rank"],
            model_config["n_heads"] * model_config["qk rope head dim"])
        weight_rk = Matrix(model_config["d_emb"],
                           model_config["qk rope head dim"])
        weight_op = Matrix(
            model_config["n_heads"] * model_config["qk nope head dim"],
            model_config["d_emb"])
        weight_gate = Matrix(model_config["d_emb"],
                             model_config["intermediate dim"])
        weight_up = Matrix(model_config["d_emb"],
                           model_config["intermediate dim"])
        weight_down = Matrix(model_config["intermediate dim"],
                             model_config["d_emb"])
        weight_router = Matrix(model_config['d_emb'], model_config['n_experts'])
        weight_gate_routed = Matrix(model_config['d_emb'],
                                    model_config['moe intermediate dim'])
        weight_up_routed = Matrix(model_config['d_emb'],
                                  model_config['moe intermediate dim'])
        weight_down_routed = Matrix(model_config['moe intermediate dim'],
                                    model_config['d_emb'])
        weight_gate_shared = Matrix(model_config['d_emb'],
                                    model_config['moe intermediate dim'])
        weight_up_shared = Matrix(model_config['d_emb'],
                                  model_config['moe intermediate dim'])
        weight_down_shared = Matrix(model_config['moe intermediate dim'],
                                    model_config['d_emb'])

        #Activation matrix
        hidden_state = Matrix(1, 1, 1)
        compressed_q = Matrix(1, 1, 1)
        decompressed_q = Matrix(1, 1, 1)
        compressed_kv = Matrix(1, 1, 1)
        decompressed_k = Matrix(1, 1, 1)
        decompressed_v = Matrix(1, 1, 1)
        ropped_k = Matrix(1, 1, 1)
        ropped_q = Matrix(1, 1, 1)
        duplicated_ropped_k = Matrix(1, 1, 1)
        concated_q = Matrix(1, 1, 1)
        concated_k = Matrix(1, 1, 1)
        scored_result = Matrix(1, 1, 1)
        mask_scale_softmax_result = Matrix(1, 1, 1)
        context_result = Matrix(1, 1, 1)
        out_proj_result = Matrix(1, 1, 1)
        residual_addition_result = Matrix(1, 1, 1)
        post_attn_norm_result = Matrix(1, 1, 1)
        gate_proj_result = Matrix(1, 1, 1)
        up_proj_result = Matrix(1, 1, 1)
        silu_result = Matrix(1, 1, 1)
        down_proj_result = Matrix(1, 1, 1)
        result_vector = Matrix(1, 1, 1)
        routed_result = Matrix(1, 1, 1)
        gate_routed_result = Matrix(1, 1, 1)
        up_routed_result = Matrix(1, 1, 1)
        silu_routed_result = Matrix(1, 1, 1)
        down_routed_result = Matrix(1, 1, 1)
        gate_shared_result = Matrix(1, 1, 1)
        up_shared_result = Matrix(1, 1, 1)
        silu_shared_result = Matrix(1, 1, 1)
        down_shared_result = Matrix(1, 1, 1)
        transposed_k_up_result = Matrix(1, 1, 1)
        score_NOPE_result = Matrix(1, 1, 1)
        score_ROPE_result = Matrix(1, 1, 1)
        v_up_context_result = Matrix(1, 1, 1)
        out_proj_context_result = Matrix(1, 1, 1)

        w_uk_first_layers = [
            Layer("pre_attn_norm", input_matrix, None, hidden_state),
            Layer("query_down", input_matrix, weight_dq, compressed_q),
            Layer("k_rope_w", hidden_state, weight_rk, ropped_k),
            Layer("kv_down", hidden_state, weight_dkv, compressed_kv),
            Layer("norm_for_compressed_q", compressed_q, None, compressed_q),
            Layer("norm_for_compressed_kv", compressed_kv, None, compressed_kv),
            Layer("q_rope_w", compressed_q, weight_rq, ropped_q),
            Layer("query_up", compressed_q, weight_uq, decompressed_q),
            Layer("transposed (k up proj)", decompressed_q, weight_uk,
                  transposed_k_up_result),
            Layer("q_rope", ropped_q, None, ropped_q),
            Layer("k_rope", ropped_k, None, ropped_k),
            Layer("score layer for RoPE", ropped_q, ropped_k,
                  score_ROPE_result),
            Layer("score layer for NoPE", transposed_k_up_result, compressed_kv,
                  score_NOPE_result),
            Layer("mask_scale_softmax", score_ROPE_result, None,
                  mask_scale_softmax_result),
            Layer("context_matmul", mask_scale_softmax_result, compressed_kv,
                  context_result),
            Layer("v_up_proj_context", context_result, weight_uv,
                  v_up_context_result),
            Layer("out_proj_context", v_up_context_result, weight_op,
                  out_proj_context_result),
            Layer("residual_addition", out_proj_context_result, None,
                  residual_addition_result),
            Layer("post_attn_norm", residual_addition_result, None,
                  post_attn_norm_result)
        ]

        w_uk_first_non_moe_ffn_layers = [
            Layer("gate_proj", post_attn_norm_result, weight_gate,
                  gate_proj_result),
            Layer("up_proj", post_attn_norm_result, weight_up, up_proj_result),
            Layer("silu", up_proj_result, None, silu_result),
            Layer("down_proj", silu_result, weight_down, down_proj_result),
            Layer("residual_addition2", down_proj_result, None, result_vector)
        ]

        w_uk_first_moe_ffn_layers = [
            Layer("gate_shared", post_attn_norm_result, weight_gate_shared,
                  gate_shared_result),
            Layer("up_shared", post_attn_norm_result, weight_up_shared,
                  up_shared_result),
            Layer("silu_shared", up_shared_result, None, silu_shared_result),
            Layer("down_shared", silu_shared_result, weight_down_shared,
                  down_shared_result),
            Layer("router", post_attn_norm_result, weight_router,
                  routed_result),
            Layer("gate_routed", post_attn_norm_result, weight_gate_routed,
                  gate_routed_result),
            Layer("up_routed", post_attn_norm_result, weight_up_routed,
                  up_routed_result),
            Layer("silu_routed", up_routed_result, None, silu_routed_result),
            Layer("down_routed", silu_routed_result, weight_down_routed,
                  down_routed_result),
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
                layer.inputA.rows = layer.inputA.rows * model_config[
                    'top-k'] / model_config['n_experts']
                if layer.inputA.rows < 1:
                    layer.inputA.rows = 1
            elif layer.name == "score layer for RoPE" and decode_flag == True:
                # layer.inputB.cols = layer.inputB.cols + input_len  + i
                layer.inputB.cols = (output_len + 1) / 2 + input_len

            result = layer.forward()
            layer.output.reshape(result)
            # print(layer.output)

            if layer.name == "score layer for RoPE":
                layer.output.rows = input_len * model_config["n_heads"]
            #reshape after q_rope
            if layer.name == "q_rope" and decode_flag == True:
                layer.output.rows = 128
                layer.output.cols = 64
                ropped_k.transpose()
            elif layer.name == "q_rope" and decode_flag == False:
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
        Matrix.reset_flops()
