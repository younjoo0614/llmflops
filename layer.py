import json
import config
from communication import get_allreduce_cost

from matrix import Matrix
from tp_type import TPType

with open("device_config.json", "r") as file:
    data = json.load(file)


class Layer:
    throughput = data["TFLOPS"]
    hbm_bw = data["HBM_BW"]
    lp_bw = data["LPDDR_BW"]
    shmem_size = data["SH_MEM"]

    def __init__(
        self,
        name,
        inputA: Matrix,
        inputB: Matrix = None,
        output: Matrix = None,
        tp: TPType = TPType.NONE,
        tp_degree: int = 1,
    ):
        self.name = name
        self.inputA = inputA
        self.inputB = inputB
        self.output = output
        self.flops = None
        self.tp = tp
        self.tp_degree = tp_degree

        if self.tp == TPType.COL:
            if self.inputB:
                self.inputB.cols = self.inputB.cols / self.tp_degree
            else:
                self.inputA.cols = self.inputA.cols / self.tp_degree
        elif self.tp == TPType.ROW:
            self.inputA.cols = self.inputA.cols / self.tp_degree
            if self.inputB:
                self.inputB.rows = self.inputB.rows / self.tp_degree
        elif self.tp == TPType.ROW_IN:
            self.inputA.rows = self.inputA.rows / 128
            self.inputA.batch = self.inputA.batch * 128 / self.tp_degree
        elif self.tp == TPType.HEAD_ROW_COL:
            self.inputB.cols = self.inputB.cols / 128
            self.inputB.batch = self.inputB.batch * (128 / self.tp_degree)
        elif self.tp == TPType.HEAD_COL_COL:
            self.inputA.cols = self.inputA.cols / 128
            self.inputA.batch = self.inputA.batch * 128 / self.tp_degree
            self.inputB.cols = self.inputB.cols / 128
            self.inputB.batch = self.inputB.batch * 128 / self.tp_degree
        else:  # tp == None
            pass

    def forward(self):
        if "norm" in self.name:
            result, self.flops = self.inputA.norm(True)

        elif "softmax" in self.name:
            result, self.flops = self.inputA.softmax(True)

        elif "rope" in self.name and "rope_w" not in self.name:
            result, self.flops = self.inputA.rope(True)

        elif "residual" in self.name:
            result, self.flops = self.inputA.residual_addition(True)

        elif "silu" in self.name:
            result, self.flops = self.inputA.silu(True)

        elif (
            "score" in self.name
            and "score layer for RoPE" != self.name
            and "score layer for NoPE" != self.name
            or "transposed" in self.name
        ):
            result, self.flops = self.inputA.score_head(self.inputB, True)

        elif (
            "context" in self.name
            and "context_matmul" != self.name
            and "out_proj" != self.name
        ):
            result, self.flops = self.inputA.context_head(self.inputB, True)

        elif "out_proj" in self.name:
            result, self.flops = self.inputA.out_proj_head(self.inputB, True)

        elif "flash_attention" in self.name:
            result, self.flops = self.inputA.flash_attention(self.inputB, True)

        elif "flash_mla" in self.name:
            result, self.flops = self.inputA.flash_mla(self.inputB, True)

        else:
            result, self.flops = self.inputA.matmul(self.inputB, True)

        return result

    def get_op_per_byte(self):
        if self.name == "flash_attention":
            # byte = 2 * self.inputB.rows * self.inputB.cols * self.inputB.data_size #2*d*N
            # byte = (
            #     byte + 3 * self.inputA.rows * self.inputA.rows
            #     + (
            #         16
            #         * self.inputA.rows
            #         * self.inputA.rows
            #         * self.inputA.cols
            #         / Layer.shmem_size
            #     ) * self.inputA.data_size
            # )
            # byte = byte + self.output.rows * self.output.cols * self.output.data_size
            # byte = byte *  self.inputA.batch

            d = self.inputA.cols
            block_c = config.SH_MEM / (4 * d * self.inputA.data_size)
            block_r = min(d, (config.SH_MEM / (4 * d * self.inputA.data_size)))
            tile_c = self.inputA.rows / block_c
            tile_r = self.inputA.rows / block_r

            byte = 2 * d * self.inputA.rows #2dN
            byte = byte + (tile_r * tile_c * (3*block_r*d + 4* block_r))
            byte = byte * self.inputA.batch * self.inputA.data_size

            print(byte)

            # byte = 2 * self.inputB.rows * self.inputB.cols #2*d*N
            # byte += ((12*self.inputA.rows*self.inputA.rows*self.inputA.cols*self.inputA.cols + 
            #           16*self.inputA.rows*self.inputA.rows*self.inputB.cols) / Layer.shmem_size)
            # byte = byte *  self.inputA.batch * self.inputA.data_size
            # print("second")
            # print(byte)

        elif self.name == "flash_mla":
            byte = (
                self.inputB.cols * 576 + (512 + 576) * (self.inputA.rows / self.tp_degree)
            )
            byte = byte * self.inputA.data_size * self.inputA.batch
        elif self.inputB is not None:
            byte = (
                self.inputA.get_size() + self.inputB.get_size() + self.output.get_size()
            )
        else:
            byte = self.inputA.get_size() + self.output.get_size()
        return self.flops / byte

    def get_flops(self):
        return int(self.flops)

    def get_execution_time(self):
        op_per_byte = self.get_op_per_byte()
        hbm_balance_point = Layer.throughput * 1024 / Layer.hbm_bw

        if op_per_byte < hbm_balance_point:
            if self.name == "flash_attention":
                # print("B: ", self.inputB)
                # print("A: ", self.inputA)

                d = self.inputA.cols
                block_c = config.SH_MEM / (4 * d * self.inputA.data_size)
                block_r = min(d, (config.SH_MEM / (4 * d * self.inputA.data_size)))
                tile_c = self.inputA.rows / block_c
                tile_r = self.inputA.rows / block_r

                byte = 2 * d * self.inputA.rows #2dN
                byte = byte + (tile_r * tile_c * (3*block_r*d + 4* block_r))
                byte = byte * self.inputA.batch * self.inputA.data_size
                
                return byte / Layer.hbm_bw / 1e3
            elif self.name == "flash_mla":
                # print(self.inputB)
                byte = (
                    self.inputB.cols * 576 + (512 + 576) * (self.inputA.rows / self.tp_degree)
                )
                byte = byte * self.inputA.data_size * self.inputA.batch
                return byte / Layer.hbm_bw / 1e3
            if self.inputB is not None:
                return (
                    (
                        self.inputA.get_size() + self.inputB.get_size() + self.output.get_size()
                    ) / Layer.hbm_bw / 1e3
                )
            else:
                return (
                    (self.inputA.get_size() + self.output.get_size()) / Layer.hbm_bw / 1e3
                )
        else:
            return (self.flops) / Layer.throughput / 1e6
