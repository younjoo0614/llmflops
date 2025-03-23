from matrix import Matrix

import json

with open("device_config.json", "r") as file:
    data = json.load(file)


class Layer:
    throughput = data["TFLOPS"]
    hbm_bw = data["HBM_BW"]
    lp_bw = data["LPDDR_BW"]

    def __init__(self, name, inputA: Matrix, inputB: Matrix = None, output: Matrix = None):
        self.name = name
        self.inputA = inputA
        self.inputB = inputB
        self.output = output
        self.flops = None

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

        elif "score" in self.name and "score layer for RoPE" != self.name and "score layer for NoPE" != self.name or "transposed" in self.name:
            result, self.flops = self.inputA.score_head(self.inputB, True)

        elif "context" in self.name and "context_matmul" != self.name and "out_proj_context" != self.name:
            result, self.flops = self.inputA.context_head(self.inputB, True)

        elif "out_proj_context" == self.name:

            result, self.flops = self.inputA.out_proj_context_head(self.inputB, True)

        else:
            result, self.flops = self.inputA.matmul(self.inputB, True)

        return result

    def get_op_per_byte(self):
        if self.inputB is not None:
            byte = self.inputA.get_size() + self.inputB.get_size() + self.output.get_size()
        else:
            byte = self.inputA.get_size() + self.output.get_size()
        return self.flops / byte

    def get_flops(self):
        return int(self.flops)

    def get_execution_time(self):
        op_per_byte = self.get_op_per_byte()
        hbm_balance_point = Layer.throughput * 1024 / Layer.hbm_bw

        if op_per_byte < hbm_balance_point:
            if self.inputB is not None:
                return (self.inputA.get_size() + self.inputB.get_size() + self.output.get_size()) / Layer.hbm_bw / 1e3
            else:
                return (self.inputA.get_size() + self.output.get_size()) / Layer.hbm_bw / 1e3
        else:
            return (self.flops) / Layer.throughput / 1e6
