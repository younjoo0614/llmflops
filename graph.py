import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches

def create_time_graph(df):
    """
    df는 "Layer Name", "OP/B", "Execution_time" 컬럼을 포함해야 합니다.
    
    - 'Layer Name'에 "residual" 또는 "norm"이 포함된 행은 제거합니다.
    - "routed"가 포함된 행은 Execution_time에 256을 곱하여 업데이트합니다.
    - Execution_time의 각 레이어 비중(%)를 계산하여 가로 길이로 사용합니다.
    - OP/B 값에 log10 변환을 적용하여 직사각형의 높이로 사용합니다.
    - 각 직사각형 내부 중앙에 레이어명과 Execution_time 비중(%)을 90도 회전하여 표시합니다.
    """
    # 'residual' 또는 'norm'이 포함된 행 제거 (대소문자 구분 없이)
    mask = ~(df['Layer Name'].str.contains('residual', case=False, na=False) |
             df['Layer Name'].str.contains('norm', case=False, na=False))
    df = df[mask].reset_index(drop=True)
    
    # "routed"가 포함된 행의 Execution_time 업데이트
    for i, row in df.iterrows():
        if 'routed' in row['Layer Name']:
            print("Debug")
            print(row['Execution_time'])
            df.loc[i, 'Execution_time'] = row['Execution_time'] * 256
            print(df.loc[i, 'Execution_time'])
    
    # Execution_time과 OP/B를 숫자로 변환
    df["Execution_time"] = pd.to_numeric(df["Execution_time"], errors="coerce")
    df["OP/B"] = pd.to_numeric(df["OP/B"], errors="coerce")

    # 총 Execution_time 합계와 각 레이어 비중 계산
    total_time = df["Execution_time"].sum()
    print("Total time", total_time)
    df["Percentage"] = (df["Execution_time"] / total_time) * 100

    # OP/B에 log10 변환 적용 (양수 값만)
    df["Transformed_OPB"] = df["OP/B"].apply(lambda x: np.log10(x) if pd.notna(x) and x > 0 else np.nan)

    # 전체 100% 기준 각 레이어의 시작 위치(left) 계산
    cumulative = np.insert(np.cumsum(df["Percentage"].values), 0, 0)
    lefts = cumulative[:-1]

    # 플롯 생성
    fig, ax = plt.subplots(figsize=(12, 6))

    # 각 레이어마다 직사각형 그리기
    for i, row in df.iterrows():
        transformed_value = row["Transformed_OPB"]
        perc = row["Percentage"]
        if pd.notna(transformed_value):
            # 직사각형 생성
            rect = patches.Rectangle((lefts[i], 0), perc, transformed_value,
                                     edgecolor='black', facecolor='skyblue')
            ax.add_patch(rect)
            # 텍스트를 직사각형 내부 중앙에 90도 회전하여 표시
            ax.text(lefts[i] + perc/2, transformed_value/2,
                    f"{row['Layer Name']}\n({row['Percentage']:.1f}%)",
                    ha='center', va='center', fontsize=8, rotation=90, color='black')
    
    # 축 범위 설정
    ax.set_xlim(0, 100)
    max_val = df["Transformed_OPB"].dropna().max()
    ax.set_ylim(0, max_val * 1.1)
    ax.set_xlabel("Execution Time Share (%)")
    ax.set_ylabel("log10(OP/B)")
    ax.set_title("Execution Time vs OP/B")
    
    plt.tight_layout()
    plt.savefig("./result/absorb_prfil_moe_input1024_batch256.png")
    plt.show()

# 사용 예시:
# df = pd.read_csv("your_data.csv")
# create_time_graph(df)
