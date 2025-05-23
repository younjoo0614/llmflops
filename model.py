import json
import pandas as pd
import config

from matrix import Matrix
from layer import Layer
from tp_type import TPType
from communication import get_allreduce_cost, get_moe_scatter_cost, get_moe_gather_cost


class Model:

    def __init__(self, name):
        self.name = name

    def base_layer(self, name, df, input_len, output_len, batch_size, data_size, tp_degree, dp_degree, model_config, decode_flag, moe_flag, fa_flag):

        # input matrix
        if decode_flag == False:  # prefill
            input_matrix = Matrix(input_len, model_config["d_emb"], batch_size)
        else:  # decode
            input_matrix = Matrix(1, model_config["d_emb"], batch_size)
            # input_matrix = Matrix((output_len+1)/2, model_config["d_emb"], batch_size)

        # weight matrix
        weight_dq = Matrix(model_config["d_emb"], model_config["q lora rank"], data_size=data_size)
        weight_dkv = Matrix(model_config["d_emb"], model_config["kv lora rank"], data_size=data_size)
        weight_uq = Matrix(model_config["q lora rank"], model_config["n_heads"] * model_config["qk nope head dim"], data_size=data_size)
        weight_uk = Matrix(model_config["kv lora rank"], model_config["n_heads"] * model_config["qk nope head dim"], data_size=data_size)
        weight_uv = Matrix(model_config["kv lora rank"], model_config["n_heads"] * model_config["qk nope head dim"], data_size=data_size)
        weight_rq = Matrix(model_config["q lora rank"], model_config["n_heads"] * model_config["qk rope head dim"], data_size=data_size)
        weight_rk = Matrix(model_config["d_emb"], model_config["qk rope head dim"], data_size=data_size)
        weight_op = Matrix(model_config["n_heads"] * model_config["qk nope head dim"], model_config["d_emb"], data_size=data_size)
        weight_gate = Matrix(model_config["d_emb"], model_config["intermediate dim"], data_size=data_size)
        weight_up = Matrix(model_config["d_emb"], model_config["intermediate dim"], data_size=data_size)
        weight_down = Matrix(model_config["intermediate dim"], model_config["d_emb"], data_size=data_size)
        weight_router = Matrix(model_config["d_emb"], model_config["n_experts"], data_size=data_size)
        weight_gate_routed = Matrix(model_config["d_emb"], model_config["moe intermediate dim"], data_size=data_size)
        weight_up_routed = Matrix(model_config["d_emb"], model_config["moe intermediate dim"], data_size=data_size)
        weight_down_routed = Matrix(model_config["moe intermediate dim"], model_config["d_emb"], data_size=data_size)
        weight_gate_shared = Matrix(model_config["d_emb"], model_config["moe intermediate dim"], data_size=data_size)
        weight_up_shared = Matrix(model_config["d_emb"], model_config["moe intermediate dim"], data_size=data_size)
        weight_down_shared = Matrix(model_config["moe intermediate dim"], model_config["d_emb"], data_size=data_size)

        # Activation matrix
        if decode_flag:
            seq_len = 1
        else:
            seq_len = input_len

        hidden_state = Matrix(seq_len, model_config["d_emb"], batch_size, data_size=data_size)
        compressed_q = Matrix(seq_len, model_config["q lora rank"], batch_size, data_size=data_size)
        decompressed_q = Matrix(seq_len, model_config["n_heads"] * model_config["qk nope head dim"], batch_size, data_size=data_size)
        compressed_kv = Matrix(seq_len, model_config["kv lora rank"], batch_size, data_size=data_size)
        decompressed_k = Matrix(seq_len, model_config["n_heads"] * model_config["qk nope head dim"], batch_size, data_size=data_size)
        decompressed_v = Matrix(seq_len, model_config["n_heads"] * model_config["qk nope head dim"], batch_size, data_size=data_size)
        ropped_k = Matrix(seq_len, model_config["qk rope head dim"], batch_size, data_size=data_size)
        ropped_q = Matrix(seq_len, model_config["n_heads"] * model_config["qk rope head dim"], batch_size, data_size=data_size)
        duplicated_ropped_k = Matrix(seq_len, model_config["n_heads"] * model_config["qk rope head dim"], batch_size, data_size=data_size)
        concated_q = Matrix(
            seq_len,
            model_config["n_heads"] * model_config["qk nope head dim"] + model_config["n_heads"] * model_config["qk rope head dim"],
            batch_size,
            data_size=data_size,
        )

        if decode_flag:
            concated_k = Matrix(
                (output_len + 1) / 2 + input_len,
                model_config["n_heads"] * model_config["qk nope head dim"] + model_config["n_heads"] * model_config["qk rope head dim"],
                batch_size,
                data_size=data_size,
            )
            scored_result = Matrix(seq_len, (output_len + 1) / 2 + input_len, batch_size, data_size=data_size)
            mask_scale_softmax_result = Matrix(seq_len, (output_len + 1) / 2 + input_len, batch_size, data_size=data_size)
            context_result = Matrix(seq_len, model_config["qk nope head dim"], batch_size, data_size=data_size)
        else:
            concated_k = Matrix(
                seq_len,
                model_config["n_heads"] * model_config["qk nope head dim"] + model_config["n_heads"] * model_config["qk rope head dim"],
                batch_size,
                data_size=data_size,
            )
            scored_result = Matrix(seq_len, seq_len, batch_size, data_size=data_size)
            mask_scale_softmax_result = Matrix(seq_len, seq_len, batch_size, data_size=data_size)
            context_result = Matrix(seq_len, model_config["qk nope head dim"], batch_size, data_size=data_size)

        out_proj_result = Matrix(seq_len, model_config["d_emb"], batch_size, data_size=data_size)
        residual_addition_result = Matrix(seq_len, model_config["d_emb"], batch_size, data_size=data_size)
        post_attn_norm_result = Matrix(seq_len, model_config["d_emb"], batch_size, data_size=data_size)
        post_attn_norm_result_shared = Matrix(seq_len, model_config["d_emb"], batch_size, data_size=data_size)
        gate_proj_result = Matrix(seq_len, model_config["intermediate dim"], batch_size, data_size=data_size)
        up_proj_result = Matrix(seq_len, model_config["intermediate dim"], batch_size, data_size=data_size)
        silu_result = Matrix(seq_len, model_config["intermediate dim"], batch_size, data_size=data_size)
        down_proj_result = Matrix(seq_len, model_config["d_emb"], batch_size, data_size=data_size)
        result_vector = Matrix(seq_len, model_config["d_emb"], batch_size, data_size=data_size)
        routed_result = Matrix(seq_len, model_config["n_experts"], batch_size, data_size=data_size)
        gate_routed_result = Matrix(
            seq_len * model_config["top-k"] / model_config["n_experts"], model_config["moe intermediate dim"], batch_size, data_size=data_size
        )
        up_routed_result = Matrix(
            seq_len * model_config["top-k"] / model_config["n_experts"], model_config["moe intermediate dim"], batch_size, data_size=data_size
        )
        silu_routed_result = Matrix(
            seq_len * model_config["top-k"] / model_config["n_experts"], model_config["moe intermediate dim"], batch_size, data_size=data_size
        )
        down_routed_result = Matrix(
            seq_len * model_config["top-k"] / model_config["n_experts"], model_config["d_emb"], batch_size, data_size=data_size
        )
        gate_shared_result = Matrix(seq_len, model_config["moe intermediate dim"], batch_size, data_size=data_size)
        up_shared_result = Matrix(seq_len, model_config["moe intermediate dim"], batch_size, data_size=data_size)
        silu_shared_result = Matrix(seq_len, model_config["moe intermediate dim"], batch_size, data_size=data_size)
        down_shared_result = Matrix(seq_len, model_config["d_emb"], batch_size, data_size=data_size)

        base_layers = [
            Layer("pre_attn_norm", input_matrix, None, hidden_state),
            Layer("query_down", input_matrix, weight_dq, compressed_q, None, tp_degree),
            Layer("attn_norm_1", compressed_q, None, compressed_q, None, tp_degree),
            Layer("query_up", compressed_q, weight_uq, decompressed_q, TPType.COL, tp_degree),
            Layer("kv_down", hidden_state, weight_dkv, compressed_kv, None, tp_degree),
            Layer("attn_norm_2", compressed_kv, None, compressed_kv, None, tp_degree),  # 확인
            Layer("k_up", compressed_kv, weight_uk, decompressed_k, TPType.COL, tp_degree),
            Layer("v_up", compressed_kv, weight_uv, decompressed_v, TPType.COL, tp_degree),
            Layer("k_rope_w", hidden_state, weight_rk, ropped_k, None, tp_degree),
            Layer("q_rope_w", compressed_q, weight_rq, ropped_q, TPType.COL, tp_degree),
            Layer("k_rope", ropped_k, None, ropped_k, None, tp_degree),
            Layer("q_rope", ropped_q, None, ropped_q, TPType.COL, tp_degree),
            (
                Layer("flash_attention", concated_q, concated_k, context_result, TPType.HEAD_COL_COL, tp_degree)
                if fa_flag and not decode_flag
                else Layer("score", concated_q, concated_k, scored_result, TPType.HEAD_COL_COL, tp_degree)
            ),
            Layer("mask_scale_softmax", scored_result, None, mask_scale_softmax_result, TPType.NONE, tp_degree),
            Layer("context_head", mask_scale_softmax_result, decompressed_v, context_result, TPType.COL, tp_degree),
            Layer("out_proj", context_result, weight_op, out_proj_result, TPType.ROW, tp_degree),
            Layer("residual_addition", out_proj_result, None, residual_addition_result, None, tp_degree),
            Layer("post_attn_norm", residual_addition_result, None, post_attn_norm_result, None, tp_degree),
        ]
        base_non_moe_ffn_layers = [
            Layer("gate_proj", post_attn_norm_result, weight_gate, gate_proj_result, TPType.COL, tp_degree),
            Layer("up_proj", post_attn_norm_result, weight_up, up_proj_result, TPType.COL, tp_degree),
            Layer("silu", up_proj_result, None, silu_result, TPType.COL, tp_degree),
            Layer("down_proj", silu_result, weight_down, down_proj_result, TPType.ROW, tp_degree),
            Layer("residual_addition2", down_proj_result, None, result_vector, None, tp_degree),
        ]

        base_moe_ffn_layers = [
            Layer("gate_shared", post_attn_norm_result_shared, weight_gate_shared, gate_shared_result, None, tp_degree),
            Layer("up_shared", post_attn_norm_result_shared, weight_up_shared, up_shared_result, None, tp_degree),
            Layer("silu_shared", up_shared_result, None, silu_shared_result, None, tp_degree),
            Layer("down_shared", silu_shared_result, weight_down_shared, down_shared_result, None, tp_degree),
            Layer("router", post_attn_norm_result, weight_router, routed_result),
            Layer("gate_routed", post_attn_norm_result, weight_gate_routed, gate_routed_result),
            Layer("up_routed", post_attn_norm_result, weight_up_routed, up_routed_result),
            Layer("silu_routed", up_routed_result, None, silu_routed_result),
            Layer("down_routed", silu_routed_result, weight_down_routed, down_routed_result),
            Layer("residual_addition2", down_shared_result, None, result_vector),
        ]

        if moe_flag == False:
            base_layers = base_layers + base_non_moe_ffn_layers
        else:
            base_layers = base_layers + base_moe_ffn_layers

        for layer in base_layers:
            if fa_flag and not decode_flag:
                if layer.name in ["score", "mask_scale_softmax", "context_head"]:
                    continue
            else:
                if layer.name == "flash_attention":
                    continue
            if (layer.name == "score" or layer.name == "context_head") and decode_flag:
                layer.inputB.rows = (output_len + 1) / 2 + input_len

            elif layer.name == "gate_routed":
                if decode_flag:
                    layer.inputA.rows = layer.inputA.rows * model_config["top-k"] * layer.inputA.batch / model_config["n_experts"] * dp_degree
                    layer.inputA.batch = 1
                    if layer.inputA.rows < 1:
                        layer.inputA.rows = 1
                else:
                    layer.inputA.rows = layer.inputA.rows * model_config["top-k"] / model_config["n_experts"] * dp_degree

            # print(layer.name)
            # print(layer.inputA)
            # print(layer.inputB)
            result = layer.forward()
            layer.output.reshape(result)
            # print(layer.output)

            if layer.name == "q_rope":
                if decode_flag == False:
                    duplicated_ropped_k = Matrix(input_len, model_config["qk rope head dim"] * model_config["n_heads"])
                    concated_q = decompressed_q.concat(ropped_q, False)
                    concated_k = decompressed_k.concat(duplicated_ropped_k, False)

                else:
                    duplicated_ropped_k = Matrix(1, model_config["qk rope head dim"] * model_config["n_heads"])
                    concated_q = decompressed_q.concat(ropped_q, False)
                    concated_k = decompressed_k.concat(duplicated_ropped_k, False)

            elif layer.name == "post_attn_norm":
                post_attn_norm_result_shared.reshape(post_attn_norm_result)

                # post_attn_norm_result_shared.batch = post_attn_norm_result_shared.batch / tp_degree

            df.loc[len(df)] = [
                layer.name,
                layer.get_flops(),
                layer.inputA.get_size(),
                layer.inputB.get_size() if layer.inputB is not None else "",
                layer.output.get_size(),
                layer.get_op_per_byte(),
                layer.get_execution_time(),
            ]

            if layer.name == "out_proj":
                if config.TP_DEGREE <= 1:
                    continue
                layer.parallelism_cost = get_allreduce_cost(layer.output.get_size())
                df.loc[len(df)] = ["Communication Cost 1", 0, "", "", "", "", layer.parallelism_cost]

            elif layer.name == "post_attn_norm":
                if (config.TP_DEGREE * config.DP_DEGREE) <= 1 and moe_flag == False:
                    continue
                layer.parallelism_cost = get_moe_scatter_cost(layer.output.get_size())
                df.loc[len(df)] = ["Communication Cost 2", 0, "", "", "", "", layer.parallelism_cost]

            elif layer.name == "residual_addition2":
                if (config.TP_DEGREE * config.DP_DEGREE) <= 1 and moe_flag == False:
                    continue
                layer.parallelism_cost = get_moe_gather_cost(layer.output.get_size())
                df.loc[len(df)] = ["Communication Cost 3", 0, "", "", "", "", layer.parallelism_cost]

        total_flops = df["FLOPS"].sum()
        df.loc[len(df)] = ["Total FLOPS", total_flops, "", "", "", "", ""]
        df["Execution_time"] = pd.to_numeric(df["Execution_time"], errors="coerce")
        total_time = df["Execution_time"].sum()
        df["Execution_time(%)"] = (df["Execution_time"] / total_time) * 100
        df["Execution_time(%)"] = df["Execution_time(%)"].round(2)
        # kv
        # total_kv_size = batch_size * (model_config['n_heads']*(input_len + output_len) * ((model_config['qk rope head dim'] + model_config['qk nope head dim']) + model_config['v head dim'])) * data_size * 61
        total_kv_size = batch_size * (input_len + output_len) * (model_config["qk rope head dim"] + model_config["kv lora rank"]) * data_size * 60
        df.loc[len(df)] = ["KV Cache", "", "", "", total_kv_size / 1024 / 1024 / 1024, "", "", ""]

        # weight
        df["InputB"] = pd.to_numeric(df["InputB"], errors="coerce")
        routed_mask = df["Layer Name"].str.contains("routed", case=False, na=False)
        df.loc[routed_mask, "InputB"] = df.loc[routed_mask, "InputB"] * 256 / (tp_degree * dp_degree)
        exclude_mask = df["Layer Name"].str.contains("score|context|flash", case=False, na=False)
        filtered_df = df[~exclude_mask]
        total_weight_sum = filtered_df["InputB"].sum()
        total_weight_sum = total_weight_sum * (57 if moe_flag else 3)
        total_weight_sum = total_weight_sum + (2 * model_config["d_emb"] * model_config["n_vocab"] * data_size)
        df.loc[len(df)] = ["Total Weight Sum", "", "", "", total_weight_sum / 1024 / 1024 / 1024, "", "", ""]
        Matrix.reset_flops()

    def w_uk_first_layer(
        self, name, df, input_len, output_len, batch_size, data_size, tp_degree, dp_degree, model_config, decode_flag, moe_flag, fmla_flag
    ):

        # input matrix
        if decode_flag == False:  # prefill
            input_matrix = Matrix(input_len, model_config["d_emb"], batch_size)
        else:  # decode
            input_matrix = Matrix(1, model_config["d_emb"], batch_size)

        # weight matrix
        weight_dq = Matrix(model_config["d_emb"], model_config["q lora rank"], data_size=data_size)
        weight_dkv = Matrix(model_config["d_emb"], model_config["kv lora rank"], data_size=data_size)
        weight_uq = Matrix(model_config["q lora rank"], model_config["n_heads"] * model_config["qk nope head dim"], data_size=data_size)
        weight_uk = Matrix(model_config["kv lora rank"], model_config["n_heads"] * model_config["qk nope head dim"], data_size=data_size)
        weight_uv = Matrix(model_config["kv lora rank"], model_config["n_heads"] * model_config["qk nope head dim"], data_size=data_size)
        weight_rq = Matrix(model_config["q lora rank"], model_config["n_heads"] * model_config["qk rope head dim"], data_size=data_size)
        weight_rk = Matrix(model_config["d_emb"], model_config["qk rope head dim"], data_size=data_size)
        weight_op = Matrix(model_config["n_heads"] * model_config["qk nope head dim"], model_config["d_emb"], data_size=data_size)
        weight_gate = Matrix(model_config["d_emb"], model_config["intermediate dim"], data_size=data_size)
        weight_up = Matrix(model_config["d_emb"], model_config["intermediate dim"], data_size=data_size)
        weight_down = Matrix(model_config["intermediate dim"], model_config["d_emb"], data_size=data_size)
        weight_router = Matrix(model_config["d_emb"], model_config["n_experts"], data_size=data_size)
        weight_gate_routed = Matrix(model_config["d_emb"], model_config["moe intermediate dim"], data_size=data_size)
        weight_up_routed = Matrix(model_config["d_emb"], model_config["moe intermediate dim"], data_size=data_size)
        weight_down_routed = Matrix(model_config["moe intermediate dim"], model_config["d_emb"], data_size=data_size)
        weight_gate_shared = Matrix(model_config["d_emb"], model_config["moe intermediate dim"], data_size=data_size)
        weight_up_shared = Matrix(model_config["d_emb"], model_config["moe intermediate dim"], data_size=data_size)
        weight_down_shared = Matrix(model_config["moe intermediate dim"], model_config["d_emb"], data_size=data_size)
        # Activation matrix
        if decode_flag:
            seq_len = 1
        else:
            seq_len = input_len

        hidden_state = Matrix(seq_len, model_config["d_emb"], batch_size, data_size=data_size)
        compressed_q = Matrix(seq_len, model_config["q lora rank"], batch_size, data_size=data_size)
        decompressed_q = Matrix(seq_len, model_config["n_heads"] * model_config["qk nope head dim"], batch_size, data_size=data_size)
        compressed_kv = Matrix(seq_len, model_config["kv lora rank"], batch_size, data_size=data_size)
        ropped_k = Matrix(seq_len, model_config["qk rope head dim"], batch_size, data_size=data_size)
        if decode_flag == False:
            ropped_q = Matrix(seq_len, model_config["n_heads"] * model_config["qk rope head dim"], batch_size, data_size=data_size)
        else:
            ropped_q = Matrix(model_config["n_heads"], model_config["qk rope head dim"], batch_size, data_size=data_size)
        mask_scale_softmax_result = Matrix(seq_len, seq_len, batch_size, data_size=data_size)

        context_result = Matrix(seq_len * model_config["n_heads"], model_config["kv lora rank"], batch_size, data_size=data_size)

        out_proj_result = Matrix(seq_len, model_config["n_heads"] * model_config["qk nope head dim"], batch_size, data_size=data_size)
        residual_addition_result = Matrix(seq_len, model_config["n_heads"] * model_config["qk nope head dim"], batch_size, data_size=data_size)
        post_attn_norm_result = Matrix(seq_len, model_config["n_heads"] * model_config["qk nope head dim"], batch_size, data_size=data_size)
        post_attn_norm_result_shared = Matrix(seq_len, model_config["n_heads"] * model_config["qk nope head dim"], batch_size, data_size=data_size)
        gate_proj_result = Matrix(seq_len, model_config["intermediate dim"], batch_size, data_size=data_size)
        up_proj_result = Matrix(seq_len, model_config["intermediate dim"], batch_size, data_size=data_size)
        silu_result = Matrix(seq_len, model_config["intermediate dim"], batch_size, data_size=data_size)
        down_proj_result = Matrix(seq_len, model_config["d_emb"], batch_size, data_size=data_size)
        result_vector = Matrix(seq_len, model_config["d_emb"], batch_size, data_size=data_size)
        routed_result = Matrix(seq_len, model_config["n_experts"], batch_size, data_size=data_size)
        gate_routed_result = Matrix(
            seq_len * model_config["top-k"] / model_config["n_experts"], model_config["moe intermediate dim"], batch_size, data_size=data_size
        )
        up_routed_result = Matrix(
            seq_len * model_config["top-k"] / model_config["n_experts"], model_config["moe intermediate dim"], batch_size, data_size=data_size
        )
        silu_routed_result = Matrix(
            seq_len * model_config["top-k"] / model_config["n_experts"], model_config["moe intermediate dim"], batch_size, data_size=data_size
        )
        down_routed_result = Matrix(
            seq_len * model_config["top-k"] / model_config["n_experts"], model_config["d_emb"], batch_size, data_size=data_size
        )
        gate_shared_result = Matrix(seq_len, model_config["moe intermediate dim"], batch_size, data_size=data_size)
        up_shared_result = Matrix(seq_len, model_config["moe intermediate dim"], batch_size, data_size=data_size)
        silu_shared_result = Matrix(seq_len, model_config["moe intermediate dim"], batch_size, data_size=data_size)
        down_shared_result = Matrix(seq_len, model_config["d_emb"], batch_size, data_size=data_size)
        transposed_k_up_result = Matrix(seq_len, model_config["kv lora rank"], batch_size * model_config["n_heads"], data_size=data_size)
        if decode_flag:
            score_NOPE_result = Matrix(seq_len * model_config["n_heads"], (output_len + 1) / 2 + input_len, batch_size, data_size=data_size)
            score_ROPE_result = Matrix(seq_len * model_config["n_heads"], (output_len + 1) / 2, batch_size, data_size=data_size)
            v_up_context_result = Matrix(seq_len, model_config["qk nope head dim"], batch_size, data_size=data_size)
            out_proj_context_result = Matrix(seq_len, model_config["d_emb"], batch_size, data_size=data_size)

        else:
            score_NOPE_result = Matrix(seq_len * model_config["n_heads"], seq_len, batch_size, data_size=data_size)
            score_ROPE_result = Matrix(seq_len * model_config["n_heads"], seq_len, batch_size, data_size=data_size)
            v_up_context_result = Matrix(seq_len, model_config["qk nope head dim"], batch_size, data_size=data_size)
            out_proj_context_result = Matrix(seq_len, model_config["d_emb"], batch_size, data_size=data_size)

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
            (
                Layer("flash_mla", transposed_k_up_result, compressed_kv, context_result, TPType.ROW_IN, tp_degree)
                if fmla_flag
                else Layer("score layer for NoPE", transposed_k_up_result, compressed_kv, score_NOPE_result, TPType.ROW_IN, tp_degree)
            ),
            Layer("score layer for RoPE", ropped_q, ropped_k, score_ROPE_result, TPType.ROW_IN, tp_degree),
            Layer("mask_scale_softmax", score_ROPE_result, None, mask_scale_softmax_result, TPType.ROW_IN, tp_degree),
            Layer("context_matmul", mask_scale_softmax_result, compressed_kv, context_result, TPType.ROW_IN, tp_degree),
            Layer("v_up_proj", context_result, weight_uv, v_up_context_result, TPType.HEAD_ROW_COL, tp_degree),
            Layer("out_proj", v_up_context_result, weight_op, out_proj_context_result, TPType.ROW, tp_degree),
            Layer("residual_addition", out_proj_context_result, None, residual_addition_result, None, tp_degree),
            Layer("post_attn_norm", residual_addition_result, None, post_attn_norm_result, None, tp_degree),
        ]

        w_uk_first_non_moe_ffn_layers = [
            Layer("gate_proj", post_attn_norm_result, weight_gate, gate_proj_result, TPType.COL, tp_degree),
            Layer("up_proj", post_attn_norm_result, weight_up, up_proj_result, TPType.COL, tp_degree),
            Layer("silu", up_proj_result, None, silu_result, TPType.COL, tp_degree),
            Layer("down_proj", silu_result, weight_down, down_proj_result, TPType.ROW, tp_degree),
            Layer("residual_addition2", down_proj_result, None, result_vector),
        ]

        w_uk_first_moe_ffn_layers = [
            Layer("gate_shared", post_attn_norm_result_shared, weight_gate_shared, gate_shared_result, None, tp_degree),
            Layer("up_shared", post_attn_norm_result_shared, weight_up_shared, up_shared_result, None, tp_degree),
            Layer("silu_shared", up_shared_result, None, silu_shared_result, None, tp_degree),
            Layer("down_shared", silu_shared_result, weight_down_shared, down_shared_result, None, tp_degree),
            Layer("router", post_attn_norm_result, weight_router, routed_result),
            Layer("gate_routed", post_attn_norm_result, weight_gate_routed, gate_routed_result),
            Layer("up_routed", post_attn_norm_result, weight_up_routed, up_routed_result),
            Layer("silu_routed", up_routed_result, None, silu_routed_result),
            Layer("down_routed", silu_routed_result, weight_down_routed, down_routed_result),
            Layer("residual_addition2", down_shared_result, None, result_vector),
        ]

        if moe_flag == False:
            w_uk_first_layers = w_uk_first_layers + w_uk_first_non_moe_ffn_layers
        else:
            w_uk_first_layers = w_uk_first_layers + w_uk_first_moe_ffn_layers

        for layer in w_uk_first_layers:
            if fmla_flag:
                if layer.name in ["score layer for RoPE", "mask_scale_softmax", "context_matmul"]:
                    continue
                if decode_flag:
                    if layer.name == "flash_mla":
                        compressed_kv.transpose()
                        compressed_kv.cols = (output_len + 1) / 2 + input_len
                        transposed_k_up_result.concat(ropped_q, row=False)
                        ropped_k.cols = (output_len + 1) / 2 + input_len
                        compressed_kv.concat(ropped_k, row=True)
                else:
                    if layer.name == "flash_mla":
                        compressed_kv.transpose()
                        transposed_k_up_result.concat(ropped_q, row=False)
                        compressed_kv.concat(ropped_k, row=True)

            elif layer.name == "flash_mla":
                continue
            if layer.name == "score layer for NoPE":
                layer.inputB.transpose()
                if decode_flag:
                    # layer.inputB.cols = layer.inputB.cols + input_len + i
                    layer.inputB.cols = (output_len + 1) / 2 + input_len
            elif layer.name == "context_matmul":
                layer.inputB.transpose()
            elif layer.name == "gate_routed":
                if decode_flag:
                    layer.inputA.rows = layer.inputA.rows * model_config["top-k"] * layer.inputA.batch / model_config["n_experts"] * dp_degree
                    layer.inputA.batch = 1
                    if layer.inputA.rows < 1:
                        layer.inputA.rows = 1
                else:
                    layer.inputA.rows = layer.inputA.rows * model_config["top-k"] / model_config["n_experts"] * dp_degree
            elif layer.name == "score layer for RoPE" and decode_flag:
                layer.inputB.cols = (output_len + 1) / 2 + input_len
            elif layer.name == "v_up_proj":
                layer.inputA.rows = layer.inputA.rows / (model_config["n_heads"] / tp_degree)
                layer.inputA.batch = layer.inputA.batch * (model_config["n_heads"] / tp_degree)

            # print(layer.name)
            # print(layer.inputA)
            # print(layer.inputB)
            result = layer.forward()
            layer.output.reshape(result)
            # print(layer.output)

            # if layer.name == "score layer for RoPE":
            # layer.output.rows = input_len * model_config["n_heads"]
            # reshape after q_rope
            if layer.name == "k_rope":
                ropped_k.transpose()
            elif layer.name == "query_up":
                layer.output.cols = layer.output.cols / model_config["n_heads"] * tp_degree
                layer.output.batch = layer.output.batch * model_config["n_heads"] / tp_degree
            elif layer.name == "q_rope_w":
                if decode_flag:
                    layer.output.rows = layer.output.cols / model_config["qk rope head dim"]
                    layer.output.cols = model_config["qk rope head dim"]
                else:
                    layer.output.rows = layer.output.rows * (model_config["n_heads"] / tp_degree)
                    layer.output.cols = model_config["qk rope head dim"]
            elif layer.name == "transposed (k up proj)":
                layer.output.rows = layer.output.rows * model_config["n_heads"] / tp_degree
                layer.output.batch = layer.output.batch / (model_config["n_heads"] / tp_degree)
            elif layer.name == "post_attn_norm":
                post_attn_norm_result_shared.reshape(post_attn_norm_result)

                # post_attn_norm_result_shared.batch = post_attn_norm_result_shared.batch / tp_degree

            df.loc[len(df)] = [
                layer.name,
                layer.get_flops(),
                layer.inputA.get_size(),
                layer.inputB.get_size() if layer.inputB is not None else "",
                layer.output.get_size(),
                layer.get_op_per_byte(),
                layer.get_execution_time(),
            ]

            if layer.name == "out_proj":
                if config.TP_DEGREE <= 1:
                    continue
                layer.parallelism_cost = get_allreduce_cost(layer.output.get_size())
                df.loc[len(df)] = ["Communication Cost 1", 0, "", "", "", "", layer.parallelism_cost]

            elif layer.name == "post_attn_norm":
                if (config.TP_DEGREE * config.DP_DEGREE) <= 1 and moe_flag == False:
                    continue
                layer.parallelism_cost = get_moe_scatter_cost(layer.output.get_size())
                df.loc[len(df)] = ["Communication Cost 2", 0, "", "", "", "", layer.parallelism_cost]

            elif layer.name == "residual_addition2":
                if (config.TP_DEGREE * config.DP_DEGREE) <= 1 and moe_flag == False:
                    continue
                layer.parallelism_cost = get_moe_gather_cost(layer.output.get_size())
                df.loc[len(df)] = ["Communication Cost 3", 0, "", "", "", "", layer.parallelism_cost]

        total_flops = df["FLOPS"].sum()
        df.loc[len(df)] = ["Total FLOPS", total_flops, "", "", "", "", ""]
        df["Execution_time"] = pd.to_numeric(df["Execution_time"], errors="coerce")
        total_time = df["Execution_time"].sum()
        df["Execution_time(%)"] = (df["Execution_time"] / total_time) * 100
        df["Execution_time(%)"] = df["Execution_time(%)"].round(2)

        # KV
        total_kv_size = batch_size * ((input_len + output_len) * (model_config["qk rope head dim"] + model_config["kv lora rank"])) * data_size * 60
        df.loc[len(df)] = ["KV Cache", "", "", "", (total_kv_size / 1024 / 1024 / 1024), "", "", ""]

        # Weight
        df["InputB"] = pd.to_numeric(df["InputB"], errors="coerce")
        routed_mask = df["Layer Name"].str.contains("routed", case=False, na=False)
        df.loc[routed_mask, "InputB"] = df.loc[routed_mask, "InputB"] * 256 / (tp_degree * dp_degree)
        exclude_mask = df["Layer Name"].str.contains("score|context|flash", case=False, na=False)
        filtered_df = df[~exclude_mask]
        total_weight_sum = filtered_df["InputB"].sum()
        total_weight_sum = total_weight_sum * (57 if moe_flag else 3)
        total_weight_sum = total_weight_sum + 2 * model_config["d_emb"] * model_config["n_vocab"] * data_size
        df.loc[len(df)] = ["Total Weight Sum", "", "", "", total_weight_sum / 1024 / 1024 / 1024, "", "", ""]
        Matrix.reset_flops()
