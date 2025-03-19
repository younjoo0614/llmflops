import argparse

DQ_D = 1536
h_d = 7168
DKV_D = 512
UH_D = 16384
NUM_HEADS = 128

DATA_SIZE = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-len', default=1024)
    parser.add_argument('--batch-size', default=1)
    parser.add_argument('--absorb', action='store_true', default=True)

    args = parser.parse_args()

    seq_len = args.seq_len
    batch = args.batch_size
    is_absorb = args.absorb

    h = Matrix(seq_len, h_d, batch)
    w_dq = Matrix(h_d, DQ_D)
    w_dkv = Matrix(h_d, DKV_D)
    w_uq = Matrix(DQ_D, UH_D)
    w_uk = Matrix(DKV_D, UH_D)
    w_uv = Matrix(DKV_D, UH_D)
    w_rq = Matrix(DQ_D, UH_D)

    ####base
    c_q, _ = h.matmul(w_dq, True) 
    c_kv, _ = h.matmul(w_dkv, True)
    u_v, _ = c_kv.matmul(w_uv, True)


    ###### example
    q_upproj = Layer("q_upproj", c_q, w_uq, True)
    u_q = q_upproj.forward()
    k_upproj = Layer("k_upproj", c_kv, w_uk, True)


    ####absorb

    print(u_v.total_flops)


if __name__ == "__main__":
    main()