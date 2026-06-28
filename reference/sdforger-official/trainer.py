# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Dict
import os
import random
import shutil
import sys

# Third Party
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import numpy as np
import pandas as pd
import torch

# Local
from fms_dgt.base.registry import register_block
from fms_dgt.constants import DATASET_TYPE
from fms_dgt.core.blocks.trainer.trainer import Trainer as TrainerBlock
from fms_dgt.core.blocks.trainer.trainer import TrainingException, get_model_dir
from fms_dgt.public.databuilders.time_series.utils import embeddings_to_text


# paper-note-added: 本类实现论文第6步流程中的第(4)步“LLM微调与推理”，对应正文 Sec3.2.2。
# paper-note-added: 即在文本编码后的嵌入系数文本上微调 GPT-2（Adam, 早停），生成阶段再自回归填充系数。
@register_block("public/trainers/sdforger-tuning")
class SDForgerTuningBlock(TrainerBlock):
    def __init__(
        self,
        seed=42,
        **kwargs,
    ):
        """Create SDForger trainer instance.

        Args:
            seed (int, optional): Seed value for training. Defaults to 42.
        """
        super().__init__(**kwargs)
        self.set_seed(seed)

    # paper-note-added: 设备选择属于框架管线，无直接论文映射；仅决定在 mps/cuda/cpu 上训练。
    @property
    def compute_device(self) -> str:
        if sys.platform == "darwin":
            return "mps"
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    # paper-note-added: 固定随机种子，保证微调/采样可复现；属于工程可复现性设置，无具体论文公式对应。
    def set_seed(self, seed):
        self.seed = seed
        os.environ["PYTHONHASHSEED"] = "0"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # paper-note-added: 4/8bit 量化配置属于工程优化（节省显存），论文未提及，与方法无关的框架插件。
    def get_quantization_config(self, dtype="float32", k_bit=None) -> BitsAndBytesConfig | None:
        if k_bit:
            if k_bit == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=dtype
                )
            elif k_bit == 8:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError("k_bit must be either 4 or 8")
        else:
            quantization_config = None

        return quantization_config

    # paper-note-added: 加载待微调的因果语言模型（论文使用 GPT-2）及其 tokenizer，对应 Sec3.2.2 的 LLM 主体。
    def get_model(self, model_id_or_path, model_args) -> AutoModel:
        """Load Huggingface model from the path.

        Args:
            model_id_or_path (str): The path to load the language model.
            model_args (dict): model args for the llm.

        Returns:
            AutoModel: Huggingface model
        """
        try:
            # paper-note-added: 论文 Sec3.2.2 指定的自回归 LLM（GPT-2），用于学习并生成嵌入系数的文本表示。
            model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                torch_dtype=model_args["dtype"],
                quantization_config=self.get_quantization_config(
                    dtype=model_args["dtype"], k_bit=model_args["k_bit"]
                ),
                trust_remote_code=model_args["trust_remote_code"],
                ignore_mismatched_sizes=model_args["ignore_mismatched_sizes"],
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_id_or_path,
                trust_remote_code=model_args["trust_remote_code"],
            )
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
            model.resize_token_embeddings(len(tokenizer))
        except Exception as err:
            raise ValueError(f"Error creating Model: {str(err)}") from err

        model.to(self.compute_device)
        return model, tokenizer

    # paper-note-added: 实现 Sec3.2.2 微调主循环：把嵌入表 E（I×K 系数）转成文本→分词→划分训练/验证→Adam 微调 GPT-2→早停取最优。
    def train(
        self,
        output_dir: str,
        dataset: DATASET_TYPE,
        model_args: Dict,
        sdforger_params: Dict,
        *args,
        iteration: int = 1,
        model_id_or_path: str | None = None,
        **kwargs,
    ):
        """Train LLM on the given dataset.

        Args:
            output_dir (str): Directory to store train results.
            dataset (DATASET_TYPE): Training dataset.
            model_args (Dict): Arguments for model initialization.
            sdforger_params (Dict): Dictionary of SDForger params.
            iteration (int): Iteration number.
            model_id_or_path (str | None): Optional model path or id.

        Returns:
            str: Path to the best saved model.
        """
        model_id_or_path = self.model_id_or_path or model_id_or_path
        if model_id_or_path is None:
            raise ValueError(
                "Must provide `model_id_or_path` during initializing SDForgerTuningBlock or when calling train()"
            )

        try:
            # Load model and tokenizer
            model_args["k_bit"] = sdforger_params.get("k_bit", None)
            model, tokenizer = self.get_model(model_id_or_path, model_args)

            # Validate and convert dataset
            if isinstance(dataset, np.ndarray):
                # assigning fantom column names
                columns = [f"column_{i+1}" for i in range(dataset.shape(1))]
                dataset = pd.DataFrame(dataset, columns=columns)
            elif not isinstance(dataset, pd.DataFrame):
                raise TypeError("Dataset must be a pandas dataframe or numpy array")

            # paper-note-added: 打乱实例行顺序，避免 LLM 学到实例间的排列偏置（与 Sec3.2.1 的随机置换 pi 思想一致，此处作用于行而非特征）。
            # Shuffle and convert to HuggingFace dataset
            dataset = dataset.sample(frac=1, random_state=self.seed)
            hf_dataset = Dataset.from_pandas(dataset)

            # paper-note-added: 第(3)步“文本编码”落地处——把每个实例的嵌入系数 e_ij 转成 fill-in-the-middle 模板文本（Sec3.2.1）。
            # paper-note-added: 具体的随机特征置换 pi 与 Input/Target [blank]/[answer] 模板逻辑在 embeddings_to_text 内部实现（utils.py）。
            # Add text column from embeddings
            # pylint: disable=no-value-for-parameter
            hf_dataset = hf_dataset.add_column(
                "data2text",
                [
                    # paper-note-added: 调用文本编码器，row 即嵌入表 E 的一行（K 个系数），输出训练用的提示文本。
                    embeddings_to_text(
                        row=row,
                        columns=dataset.columns.to_list().copy(),
                        eos_token=tokenizer.eos_token,
                        input_tokens_precision=sdforger_params["input_tokens_precision"],
                    )
                    for row in tqdm(hf_dataset, desc="Generating text from embeddings")
                ],
            )

            # paper-note-added: 分词并以 input_ids 作为 labels，即标准自回归语言建模目标；属于训练管线实现，无独立论文公式。
            # Tokenization
            def preprocess_function(row):
                model_inputs = tokenizer(row["data2text"], padding=True)
                model_inputs["labels"] = model_inputs["input_ids"]
                return model_inputs

            tokenized_dataset = hf_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=hf_dataset.column_names,
            )

            # paper-note-added: Sec3.2.2 规定取 20% 作为验证集（用于早停监控 val loss），此处 test_size=0.2 与之对应。
            # Split into train (80%) and validation (20%)
            split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=self.seed)
            train_data = split_dataset["train"]
            val_data = split_dataset["test"]

            # paper-note-added: 训练超参对应 Sec3.2.2（论文设定 lr=8e-5、batch=32、最多 200 epoch）；此处默认值较保守，实际由 config 覆盖。
            # Training arguments
            train_args = {
                "learning_rate": self.training_args.get("learning_rate", 5e-5),
                "num_train_epochs": self.training_args.get("num_train_epochs", 2),
                "per_device_train_batch_size": self.training_args.get(
                    "per_device_train_batch_size", 4
                ),
            }

            # tuned model path
            tuned_model_path = os.path.join(get_model_dir(output_dir) + f"_iter-{iteration}")
            if os.path.exists(
                tuned_model_path
            ):  # Delete what's already present in the model directory
                for file_name in os.listdir(tuned_model_path):
                    file_path = os.path.join(tuned_model_path, file_name)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)  # Remove the file
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove the directory
                self.logger.info(
                    "Model files were found at %s. All files have been deleted, and the new model will replace the old one.",
                    tuned_model_path,
                )

            # paper-note-added: HF Trainer 装配微调器；adam_epsilon 对应 Sec3.2.2 使用的 Adam 优化器。
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=tuned_model_path,
                    **train_args,
                    # paper-note-added: 使用 Adam 优化器（论文 Sec3.2.2 指定）。
                    adam_epsilon=1e-04,
                    logging_strategy="steps",
                    logging_steps=10,
                    # paper-note-added: 每 5 步在验证集评估一次，对应 Sec3.2.2“每 5 步检查验证损失”的早停监控频率。
                    eval_strategy="steps",
                    eval_steps=5,
                    save_strategy="steps",
                    save_steps=100,
                    # paper-note-added: 训练结束加载验证损失最优的检查点，即早停所保留的最佳模型（Sec3.2.2）。
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                ),
                processing_class=tokenizer,
                train_dataset=train_data,
                eval_dataset=val_data,
                # paper-note-added: 早停回调 patience=5，与 Sec3.2.2 的“耐心值 5”一致。
                callbacks=[EvalLossEarlyStopping(patience=5)],
            )

            # paper-note-added: 启动 Sec3.2.2 微调循环（注意：本文件只负责微调；多项式采样/温度采样的推理生成在生成模块实现）。
            trainer.train()

            # Save best model
            best_model_path = os.path.join(tuned_model_path, "best")
            trainer.save_model(best_model_path)

        except Exception as err:
            raise TrainingException(f"Finetuning LLM failed: {str(err)}") from err

        return best_model_path


# paper-note-added: 实现 Sec3.2.2 的早停机制：监控验证损失，连续 patience 次未改善则停止训练，防止过拟合并减少不稳定。
class EvalLossEarlyStopping(TrainerCallback):
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = None
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return control

        # paper-note-added: 验证损失刷新最优则重置计数器；否则累加，达到 patience 即触发早停（Sec3.2.2）。
        if self.best_loss is None or eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                control.should_training_stop = True

        return control
