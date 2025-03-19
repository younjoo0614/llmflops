class Layer:
    def __init__(self, name, inputA, inputB):
        self.name = name
        self.inputA = inputA
        self.inputB = inputB
        self.flops = None

    def forward(self):
        # if "norm" in self.name 이나 뭐 in list 처리하든 해서 필요한 matrix function
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

        else:
            result, self.flops = self.inputA.matmul(self.inputB, True)
        
        return result
    
    def get_op_per_byte(self):
        
        return 