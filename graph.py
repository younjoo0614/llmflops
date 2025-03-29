import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches

def create_multiple_time_graph(df, name, ax, input_len=None, output_len=None, batch_size=None, tensor_parallel=None, data_parallel=None):
    # 'residual' 또는 'norm'이 포함된 행 제거 (대소문자 구분 없이)
    mask = ~(df['Layer Name'].str.contains('residual', case=False, na=False) |
             df['Layer Name'].str.contains('norm', case=False, na=False))
    df = df[mask].reset_index(drop=True)

    # 'routed'가 포함된 행의 Execution_time 업데이트
    for i, row in df.iterrows():
        if 'routed' in row['Layer Name'].lower():
            df.loc[i, 'Execution_time'] = row['Execution_time'] * (256 / (tensor_parallel * data_parallel))

    # Execution_time과 OP/B를 숫자로 변환
    df["Execution_time"] = pd.to_numeric(df["Execution_time"], errors="coerce")
    df["OP/B"] = pd.to_numeric(df["OP/B"], errors="coerce")

    # Execution_time의 총합과 각 레이어 비중 계산
    total_time = df["Execution_time"].sum()
    df["Percentage"] = (df["Execution_time"] / total_time) * 100

    # OP/B에 log10 변환 적용 (양수 값만)
    df["Transformed_OPB"] = df["OP/B"].apply(lambda x: np.log10(x) if pd.notna(x) and x > 0 else np.nan)

    # 계열별 색상 분류 함수
    def get_color(layer_name):
        lname = layer_name.lower()

        if 'residual' in lname or 'norm' in lname:
            return 'firebrick'  # 빨간색 (사실상 제거됨)

        # MoE FFN 계열
        moe_keywords = ['gate_shared', 'up_shared', 'silu_shared', 'down_shared',
                        'router', 'gate_routed', 'up_routed', 'silu_routed', 'down_routed']
        if any(k in lname for k in moe_keywords):
            return 'mediumseagreen'

        # Dense FFN 계열
        ffn_keywords = ['gate_proj', 'up_proj', 'silu', 'down_proj']
        if any(k in lname for k in ffn_keywords) and lname != 'v_up_proj_context':
            return 'salmon'

        # Attention 계열
        attn_keywords = [
            'pre_attn_norm', 'query_down', 'query_up', 'kv_down', 'k_up', 'v_up',
            'k_rope', 'q_rope', 'rope', 'score', 'mask', 'context',
            'out_proj', 'residual_addition', 'post_attn_norm', "flash_attention",
            'transposed', 'score layer', 'context_matmul', 'v_up_proj_context',
            "score layer for RoPE", "score layer for NoPE", "mask_scale_softmax"
        ]
        if any(k in lname for k in attn_keywords):
            return 'skyblue'

        return 'lightgrey'

    # 각 레이어의 시작 위치(left)를 계산 (전체 100% 기준)
    cumulative = np.insert(np.cumsum(df["Percentage"].values), 0, 0)
    lefts = cumulative[:-1]

    # Y축 설정
    ax.set_xlim(0, 100)
    if "decode" in name:
        min_tick = -1
        max_tick = int(np.ceil(df["Transformed_OPB"].dropna().max()))
        yticks = list(range(min_tick, max_tick + 1))
        ax.set_ylim(min_tick, max_tick)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{10**t:.1f}" if t < 3 else f"{10**t:.0f}" for t in yticks])
    else:
        max_tick = int(np.ceil(df["Transformed_OPB"].dropna().max()))
        yticks = list(range(0, max_tick + 1))
        ax.set_ylim(0, max_tick)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{10**t:.0f}" for t in yticks])

    ymin = ax.get_ylim()[0]

    for i, row in df.iterrows():
        transformed_value = row["Transformed_OPB"]
        perc = row["Percentage"]
        exec_time_us = row["Execution_time"]
        color = get_color(row["Layer Name"])

        is_comm_cost = "communication cost" in row["Layer Name"].lower()

        if is_comm_cost:
            # Communication Cost: OP/B 최대로 설정, 색상 강조
            transformed_value = max_tick
            color = 'orange'
            hatch = None
        else:
            hatch = None

        if pd.notna(transformed_value) and pd.notna(exec_time_us):
            height = transformed_value - ymin
            rect = patches.Rectangle((lefts[i], ymin), perc, height,
                                     edgecolor='black', facecolor=color, hatch=hatch)
            ax.add_patch(rect)

            exec_time_ms = exec_time_us / 1000
            label_text = f"{row['Layer Name']}\n({perc:.1f}%, {exec_time_ms:.2f}ms)"

            # 텍스트 항상 막대 내부 출력
            if perc >= 6:
                ax.text(lefts[i] + perc / 2, ymin + height * 0.95,
                        label_text, ha='center', va='top', fontsize=12, rotation=0, color='black')
            elif perc >= 2:
                ax.text(lefts[i] + perc / 2, ymin + height / 2,
                        label_text, ha='center', va='center', fontsize=12, rotation=90, color='black')

    # Balance point 라인
    balance_point_log = np.log10(295)
    ax.axhline(y=balance_point_log, color='red', linestyle='--', linewidth=1.5)
    ax.text(100, balance_point_log, 'Balance point', va='center', ha='right',
            fontsize=11, color='red', fontweight='bold')

    # 축 및 제목
    ax.set_xlabel("Execution Time Share (%)", fontsize=12)
    ax.set_ylabel("OP/B", fontsize=12)
    ax.set_title(f"Execution Time vs OP/B: {name}", fontsize=14)

    # 범례
    legend_elements = [
        patches.Patch(facecolor='skyblue', edgecolor='black', label='Attention'),
        patches.Patch(facecolor='salmon', edgecolor='black', label='Dense FFN'),
        patches.Patch(facecolor='mediumseagreen', edgecolor='black', label='MoE FFN'),
        patches.Patch(facecolor='lightgrey', edgecolor='black', label='Other'),
        patches.Patch(facecolor='orange', edgecolor='black', label='Communication Cost')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    kv_row = df[df["Layer Name"] == "KV Cache"]
    if not kv_row.empty:
        kv_size_bytes = pd.to_numeric(kv_row["Output"].values[0], errors="coerce")
    total_time_ms = total_time / 1000  # μs → ms
    
    weight_row = df[df["Layer Name"] == "Total Weight Sum"]
    if not weight_row.empty:
        total_weight = pd.to_numeric(weight_row["Output"].values[0], errors="coerce")

    total_time_ms = total_time / 1000  # μs → ms


    config_text = (
        f"Input: {input_len}, Output: {output_len}, Batch: {batch_size}\n"
        f"TP: {tensor_parallel}, DP: {data_parallel}, Total Time: {total_time_ms:.2f} ms\n"
        f"KV Cache: {kv_size_bytes:.2f} GB \n Total Weight Sum: {total_weight:.2f} GB"
    )

    # 그래프에 추가
    ax.text(50, max_tick + 0.3, config_text,
            ha='center', va='bottom', fontsize=11,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.subplots_adjust(top=0.82, bottom=0.12)
