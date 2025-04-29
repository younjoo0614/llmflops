import config


class Matrix:
    total_flops = 0

    def __init__(self, rows: int, cols: int, batch: int = 1, data_size: int = 2):
        self.rows = rows
        self.cols = cols
        self.batch = batch
        self.data_size = data_size

    def __str__(self):
        return f"{int(self.rows)},{int(self.cols)},{int(self.batch)}"

    @classmethod
    def reset_flops(cls):
        cls.total_flops = 0

    def get_size(self):
        return int(self.data_size * self.rows * self.cols * self.batch)

    def matmul(self, B, real):
        if self.cols != B.rows:
            raise ValueError(f"Dimension does not match self.cols: {self.cols}, B.rows: {B.rows}")
        result = Matrix(self.rows, B.cols, self.batch)
        flops = 2 * self.batch * self.rows * B.rows * B.cols
        if real:
            Matrix.total_flops = int(Matrix.total_flops) + int(flops)
        return result, flops

    def flash_attention(self, B, real):  # K cache and V cache has the same shape
        result = Matrix(self.rows, B.cols - 64, self.batch)
        flops = (
            4 * self.rows * self.rows * self.cols
            + 16 * self.rows * self.rows
            + 24 * self.rows * self.rows * self.cols / config.SH_MEM
        ) * self.batch  # + 24 row^2 * col / 256K

        if real:
            Matrix.total_flops = int(Matrix.total_flops) + flops
        return result, flops

    def flash_mla(self, B, real):
        result = Matrix(self.rows, self.cols - 64, self.batch)
        flops = self.batch * self.rows * B.cols * (self.cols + B.rows - 64) * 2
        if real:
            Matrix.flops = int(Matrix.total_flops) + flops

        return result, flops

    def score_head(self, B, real):
        B.transpose()
        if self.cols != B.rows:
            raise ValueError(f"Dimension does not match self.cols: {self.cols}, B.rows: {B.rows}")
        result = Matrix(self.rows, B.cols, self.batch)

        flops = 2 * self.rows * self.cols * result.cols * result.batch
        if real:
            Matrix.total_flops = int(Matrix.total_flops) + int(flops)
        return result, flops

    def context_head(self, B, real):
        B.cols = B.cols / (config.NUM_HEADS / config.TP_DEGREE)
        B.batch = B.batch * (config.NUM_HEADS / config.TP_DEGREE)
        result = Matrix(self.rows, B.cols, self.batch)
        flops = 2 * self.rows * self.cols * result.cols * result.batch

        if real:
            Matrix.total_flops = int(Matrix.total_flops) + int(flops)

        return result, flops

    def out_proj_head(self, B, real):
        self.cols = self.cols * config.NUM_HEADS / config.TP_DEGREE
        self.batch = self.batch / (config.NUM_HEADS / config.TP_DEGREE)
        B.rows = B.rows / config.TP_DEGREE
        B.batch = B.batch * config.TP_DEGREE
        result = Matrix(self.rows, B.cols, self.batch)
        # print(result)
        flops = (
            2 * self.rows * self.cols * result.cols * result.batch + (config.TP_DEGREE - 1) * result.rows * result.cols
        )

        if real:
            Matrix.total_flops = int(Matrix.total_flops) + int(flops)

        return result, flops

    def norm(self, real):
        flops = self.rows * self.cols * self.batch * 4
        if real:
            Matrix.total_flops = int(Matrix.total_flops) + int(flops)
        result = Matrix(self.rows, self.cols, self.batch)
        return result, flops

    def residual_addition(self, real):
        flops = self.rows * self.cols * self.batch * 1  # need to add shared experts
        if real:
            Matrix.total_flops = int(Matrix.total_flops) + int(flops)
        return self, flops

    def silu(self, real):
        flops = self.rows * self.cols * self.batch * 7
        if real:
            Matrix.total_flops = int(Matrix.total_flops) + int(flops)
        return self, flops

    def rope(self, real):
        flops = self.rows * self.cols * self.batch * 6
        if real:
            Matrix.total_flops = int(Matrix.total_flops) + int(flops)
        return self, flops

    def softmax(self, real):
        flops = self.rows * self.cols * self.batch * 7
        if real:
            Matrix.total_flops = int(Matrix.total_flops) + int(flops)
        return self, flops

    def transpose(self):
        old_col = self.cols
        self.cols = self.rows
        self.rows = old_col

    def concat(self, B, row):
        if row:
            if self.cols != B.cols:
                raise ValueError("Dimension does not match")
            self.rows = self.rows + B.rows
        else:
            if self.rows != B.rows:
                raise ValueError("Dimension does not match")
            self.cols = self.cols + B.cols
        return self

    def reshape(self, new_matrix):
        self.rows = new_matrix.rows
        self.cols = new_matrix.cols
        self.batch = new_matrix.batch
