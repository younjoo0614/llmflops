from enum import Enum, auto

class TPType(Enum):
    COL = auto()
    ROW = auto()
    HEAD_ROW_COL = auto()
    HEAD_COL_COL = auto()
    HEAD_COL_ROW = auto()
    HEAD_ROW = auto()
    HEAD_COL = auto()
    ROW_IN = auto()
    NONE = auto()
