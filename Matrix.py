class Matrix:
    total_flops = 0
    def __init__(self, rows: int, cols: int, batch: int = 1):
        self.rows = rows
        self.cols = cols
        self.batch = batch
    
    def __str__(self):
        return f"rows: {self.rows}, cols: {self.cols}, batch: {self.batch}"


    def get_size(self):
        return DATA_SIZE * self.rows * self.cols * self.batch
    
    def matmul(self, B, real):
        if (self.cols != B.rows): 
            raise ValueError("Dimension does not match")
        result = Matrix(self.rows, B.cols, self.batch)
        flops = 2 * self.batch * self.rows * B.rows * B.cols
        if real: Matrix.total_flops = int(Matrix.total_flops) + int(flops)
        return result, flops
    
    def score_head(self, B, real):
        B.transpose()
        # B.rows = B.rows + k_rope.cols * NUM_HEADS
        if (self.cols != B.rows): 
            raise ValueError("Dimension does not match")
        # result = Matrix(self.rows, B.cols, self.batch * NUM_HEADS) //need edit
        result = Matrix(self.rows, B.cols, self.batch * 128)

        # flops = 2 * self.rows * self.cols / NUM_HEADS * result.cols * result.batch
        flops = 2 * self.rows * self.cols / 128 * result.cols * result.batch

        if real: Matrix.total_flops = int(Matrix.total_flops) + int(flops)

        return result, flops

    def context_head(self, B, real):
        # result = Matrix(self.rows, B.cols / NUM_HEADS, self.batch * NUM_HEADS)
        result = Matrix(self.rows, B.cols / 128, self.batch)
        flops = 2 * self.rows * self.cols * result.cols * result.batch

        if real: Matrix.total_flops = int(Matrix.total_flops) + int(flops)

        return result, flops 

    def norm(self, real):
        flops = self.rows * self.cols * self.batch * 4
        if real: Matrix.total_flops = int(Matrix.total_flops) + int(flops)
        result = Matrix(self.rows, self.cols, self.batch)
        return result, flops

    def residual_addition(self, real):
        flops = self.rows * self.cols * self.batch * 1
        if real: Matrix.total_flops = int(Matrix.total_flops) + int(flops)
        return self, flops

    def silu (self, real):
        flops = self.rows * self.cols * self.batch * 7
        if real: Matrix.total_flops = int(Matrix.total_flops) + int(flops)
        return self, flops

    def rope(self, real):
        flops = self.rows * self.cols * self.batch * 6
        if real: Matrix.total_flops = int(Matrix.total_flops) + int(flops)
        return self, flops

    def softmax(self, real):
        flops = self.rows * self.cols * self.batch * 7
        if real: Matrix.total_flops = int(Matrix.total_flops) + int(flops)
        return self, flops

    def transpose(self):
        old_col = self.cols
        self.cols = self.rows
        self.rows = old_col
    
    def concat(self, B, row):
        if row:
            if (self.cols != B.cols):
                raise ValueError("Dimension does not match")

            self.rows = self.rows + B.rows
        else:
            if (self.rows != B.rows):
                raise ValueError("Dimension does not match")
            self.cols = self.cols + B.cols
        return self