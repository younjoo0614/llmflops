class Layer:
    def __init__(self, name, inputA, inputB=None, output=None):
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

        elif "score" in self.name and "score layer for RoPE" != self.name  and "score layer for NoPE" != self.name or "transposed" in self.name :
            result, self.flops = self.inputA.score_head(self.inputB, True)
        
        elif "context" in self.name  and "context_matmul" != self.name and "out_proj_context" != self.name:
            result, self.flops = self.inputA.context_head(self.inputB, True)
        
        elif "out_proj_context" == self.name:

            result, self.flops = self.inputA.out_proj_context_head(self.inputB, True)
            
        else:
            result, self.flops = self.inputA.matmul(self.inputB, True)
            
        return result
    
    def get_op_per_byte(self):
        
        return 
