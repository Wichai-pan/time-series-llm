# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass

# Local
from fms_dgt.base.task import DataPoint


# paper-note-added: 输入数据结构（框架管线，无直接论文公式映射）。承载 SDForger 管线的“种子/输入”原始时间序列，
# paper-note-added: 即论文整体方法的入口：原始时间序列 X 在被切分(步骤1)、嵌入(步骤2)之前先封装为此对象。
@dataclass(kw_only=True)
class TimeSeriesInputData(DataPoint):
    """This class is intended to hold the seed / machine generated instruction data"""

    task_description: str = ""
    # paper-note-added: observations 存放原始观测时间序列数据（多通道），对应论文输入 X（长度 L0，含多个 channel）。
    observations: dict = ""


# paper-note-added: 输出数据结构（框架管线，无直接论文公式映射）。承载管线末端的“生成结果”，即步骤(6)ICA/FPC 解码后
# paper-note-added: 重建出的合成时间序列 x_i = Σ_j e_ij·b_j（论文 Sec3.3 线性组合解码的最终产物）。
@dataclass(kw_only=True)
class TimeSeriesOutputData(DataPoint):
    """This class is intended to hold the seed / machine generated instruction data"""

    task_description: str = ""
    # paper-note-added: generated_time_series 存放解码后生成的合成时间序列（多通道），对应论文步骤6输出的合成样本集合。
    generated_time_series: dict = ""
