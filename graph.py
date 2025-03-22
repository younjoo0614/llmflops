import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches

def create_time_graph(df, name):
    # 'residual' 또는 'norm'이 포함된 행 제거 (대소문자 구분 없이)
    mask = ~(df['Layer Name'].str.contains('residual', case=False, na=False) |
             df['Layer Name'].str.contains('norm', case=False, na=False))
    df = df[mask].reset_index(drop=True)
    
    # 'routed'가 포함된 행의 Execution_time 업데이트
    for i, row in df.iterrows():
        if 'routed' in row['Layer Name']:
            df.loc[i, 'Execution_time'] = row['Execution_time'] * 256
    
    # Execution_time과 OP/B를 숫자로 변환
    df["Execution_time"] = pd.to_numeric(df["Execution_time"], errors="coerce")
    df["OP/B"] = pd.to_numeric(df["OP/B"], errors="coerce")
    
    # Execution_time의 총합과 각 레이어 비중 계산
    total_time = df["Execution_time"].sum()
    df["Percentage"] = (df["Execution_time"] / total_time) * 100

    # OP/B에 log10 변환 적용 (양수 값만)
    df["Transformed_OPB"] = df["OP/B"].apply(lambda x: np.log10(x) if pd.notna(x) and x > 0 else np.nan)

    # 각 레이어의 시작 위치(left)를 계산 (전체 100% 기준)
    cumulative = np.insert(np.cumsum(df["Percentage"].values), 0, 0)
    lefts = cumulative[:-1]

    # 그래프 크기를 키움
    fig, ax = plt.subplots(figsize=(20, 8))

    # 각 레이어마다 직사각형(막대) 그리기 및 텍스트 추가
    for i, row in df.iterrows():
        transformed_value = row["Transformed_OPB"]
        perc = row["Percentage"]
        if pd.notna(transformed_value):
            rect = patches.Rectangle((lefts[i], 0), perc, transformed_value,
                                     edgecolor='black', facecolor='skyblue')
            ax.add_patch(rect)
            
            # 막대 폭(perc)이 너무 작으면 텍스트 생략
            if perc >= 2:  # 예: 2% 미만이면 표시 안 함
                ax.text(lefts[i] + perc/2, transformed_value/2,
                        f"{row['Layer Name']} ({row['Percentage']:.1f}%)",
                        ha='center', va='center', fontsize=10, rotation=90, color='black')
    
    ax.set_xlim(0, 100)
    # y축: 값은 log10 변환값이므로, 티크를 0, 1, 2, ... 로 하고 레이블은 10^0, 10^1, ... 로 표현
    max_tick = int(np.ceil(df["Transformed_OPB"].dropna().max()))
    yticks = list(range(0, max_tick + 1))
    ax.set_yticks(yticks)
    ytick_labels = [f"{10**t:.0f}" for t in yticks]
    ax.set_yticklabels(ytick_labels)
    ax.set_ylim(0, max_tick)
    
    ax.set_xlabel("Execution Time Share (%)", fontsize=12)
    ax.set_ylabel("OP/B", fontsize=12)
    ax.set_title(f"Execution Time vs OP/B: {name}", fontsize=14)
    
    plt.tight_layout()
    plt.savefig("./result" + name + ".png")
    plt.show()

# 사용 예시:
# df = pd.read_csv("your_data.csv")
# create_time_graph(df, "example_graph")
