# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, Mapping, Union

# Local
from fms_dgt.base.task import TransformationTask, TransformationTaskRunnerConfig
from fms_dgt.public.databuilders.time_series.data_objects import (
    TimeSeriesInputData,
    TimeSeriesOutputData,
)
from fms_dgt.utils import init_dataclass_from_dict


# paper-note-added: 任务定义类，把 SDForger 整个 6 步管线接入 fms-dgt 框架（属框架装配/plumbing，非论文具体算法）。
# paper-note-added: 它以“变换任务(TransformationTask)”形式声明：输入=原始时间序列，输出=合成时间序列，
# paper-note-added: 并通过 data_params / sdforger_params 把论文超参（嵌入维度 k、LLM、采样温度等）传递给下游 block。
# NOTE: this class holds the information needed for the overall time series generation task
class TimeSeriesTask(TransformationTask):

    # paper-note-added: 显式声明输入/输出数据类型，对应论文管线的入口(原始 X)与出口(合成时间序列)。
    # We must always specify both the type of data that will be accepted as well as the type of data that will be generated
    INPUT_DATA_TYPE = TimeSeriesInputData
    OUTPUT_DATA_TYPE = TimeSeriesOutputData

    def __init__(
        self,
        *args: Any,
        runner_config: Union[Mapping, TransformationTaskRunnerConfig] = None,
        data_params: Dict[str, Any],
        sdforger_params: Dict[str, Any],
        **kwargs: Any,
    ):
        runner_config = init_dataclass_from_dict(runner_config, TransformationTaskRunnerConfig)
        # paper-note-added: 把一次变换的批大小设为 train_length（默认5000），即用于切分/嵌入的原始序列长度 L0（论文 A.1 分段输入长度）。
        runner_config.transform_batch_size = data_params.get("train_length", 5000)
        # paper-note-added: data_params 承载数据侧超参（如 train_length=L0、分段相关设置），对应论文 A.1 周期感知分段所需参数。
        self.data_params = data_params
        # paper-note-added: sdforger_params 承载方法侧超参（嵌入基类型 FPC/FastICA、嵌入维度 k、过滤/停止阈值等），对应论文 Sec3.1、A.2、A.3、A.4。
        self.sdforger_params = sdforger_params
        super().__init__(*args, runner_config=runner_config, **kwargs)

    # paper-note-added: 把传入的原始观测数据封装成输入样本对象，作为整个 6 步管线的入口；本身是框架装配逻辑，无具体论文公式。
    def instantiate_input_example(self, **kwargs: Any):
        return self.INPUT_DATA_TYPE(
            task_name=self.name,
            task_description=self.task_description,
            # paper-note-added: kwargs 即原始时间序列观测，对应论文输入 X（待切分为约 I=15 个窗口实例并做 ICA/FPC 嵌入）。
            observations=kwargs,
        )
