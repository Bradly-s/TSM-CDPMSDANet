# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
from typing import List

import paddle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

from paddlevideo.utils import get_logger

from .base import BaseMetric
from .registry import METRIC
import paddle.nn.functional as F

logger = get_logger("paddlevideo")


@METRIC.register
class CenterCropMetric(BaseMetric):
    def __init__(self, data_size, batch_size, log_interval=1, **kwargs):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, log_interval, **kwargs)
        self.rest_data_size = data_size  # Number of samples remaining to be tested
        self.all_outputs = []
        self.all_labels = []
        self.topk = kwargs.get("topk", [1, 5])

    def update(self, batch_id: int, data: List, outputs: paddle.Tensor) -> None:
        """update metrics during each iter

        Args:
            batch_id (int): iter id of current batch.
            data (List): list of batched data, such as [inputs, labels]
            outputs (paddle.Tensor): batched outputs from model
        """
        labels = data[1]
        if self.world_size > 1:
            labels_gathered = self.gather_from_gpu(labels, concat_axis=0)
            outpus_gathered = self.gather_from_gpu(outputs, concat_axis=0)
        else:
            labels_gathered = labels
            outpus_gathered = outputs

        # Avoid resampling effects when testing with multiple cards
        labels_gathered = labels_gathered[0:min(len(labels_gathered), self.
                                                rest_data_size)]
        outpus_gathered = outpus_gathered[0:min(len(outpus_gathered), self.
                                                rest_data_size)]
        self.all_labels.append(labels_gathered)
        self.all_outputs.append(outpus_gathered)
        self.rest_data_size -= outpus_gathered.shape[0]

        # preds ensemble
        if batch_id % self.log_interval == 0:
            logger.info("[TEST] Processing batch {}/{} ...".format(batch_id, self.data_size // (self.batch_size * self.world_size)))
            # print("self.all_outputs:", self.all_outputs)
            # print("self.all_labels:", self.all_labels)


    # 修改
    def accumulate(self):
        """accumulate, compute, and show metrics when finished all iters.
        """
        self.all_outputs = paddle.concat(self.all_outputs, axis=0)
        self.all_labels = paddle.concat(self.all_labels, axis=0)

        # Apply softmax to convert outputs to probabilities
        probabilities = F.softmax(self.all_outputs, axis=1)

        # Convert tensor to numpy for f1_score calculation
        predict_labels = paddle.argmax(probabilities, axis=1).numpy()
        true_labels = self.all_labels.numpy()

        # Calculate the F1 score
        f1 = f1_score(true_labels, predict_labels, average='weighted')
        logger.info(f"F1 score: {f1:.2f}")

        # Calculate per-class accuracy
        num_classes = probabilities.shape[1]
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        # Go through each prediction and label
        for i in range(len(predict_labels)):
            label = int(true_labels[i])  # 转换为Python整数
            class_total[label] += 1
            if predict_labels[i] == label:
                class_correct[label] += 1

        # 准备收集每个类别的准确率数据
        accuracies = []
        categories = []
        # 准备一个字典来收集每个类别的准确率
        accuracy_data = {}

        # Print per-class accuracies
        for i in range(num_classes):
            if class_total[i] > 0:
                accuracy = (class_correct[i] / class_total[i]) * 100
                logger.info(f'Accuracy of class {i} : {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})')
                accuracy_data[f'Class {i}'] = str(accuracy)
            else:
                logger.info(f'Accuracy of class {i} : N/A (no predictions)')
                accuracy_data[f'Class {i}'] = 'N/A'

        # 将数据转换成DataFrame，这里的数据是横向的，即每个类别对应一列
        df_accuracies = pd.DataFrame([accuracy_data])

        # 指定Excel文件保存路径
        # save_path = "/mnt/sdb1/swf/project/PaddleVideo/output/PPTSM_RESULTS/visul/ucf24"
        # save_path = "/mnt/sdb1/swf/project/PaddleVideo/output/PPTSM_RESULTS/visul/hmdb51"
        save_path = "/mnt/sdb1/swf/project/PaddleVideo/output/PPTSM_RESULTS/visul/hmdb51"
        # save_path = "/mnt/sdb1/swf/project/PaddleVideo/output/PPTSM_RESULTS/visul/pptsm_modify"
        save_filename = "class_accuracies.xlsx"
        full_path = os.path.join(save_path, save_filename)

        # 确保保存路径存在
        os.makedirs(save_path, exist_ok=True)

        # 检查文件是否存在，如果存在则追加数据
        if os.path.exists(full_path):
            # 读取现有Excel文件
            old_df = pd.read_excel(full_path)
            # 追加新数据
            new_df = pd.concat([old_df, df_accuracies], ignore_index=True)
        else:
            # 如果文件不存在，新DataFrame就是第一个数据集
            new_df = df_accuracies

        # 将新的或者更新的DataFrame保存为Excel文件
        new_df.to_excel(full_path, index=False)


        # Calculate and print top-k accuracy
        for k in self.topk:
            topk_accuracy = paddle.metric.accuracy(self.all_outputs, self.all_labels, k=k)
            # 如果topk_accuracy是0维的，使用.item()来获取它的标量值
            topk_accuracy_value = topk_accuracy.item() if topk_accuracy.ndim == 0 else topk_accuracy.numpy()[0]
            logger.info(f'Top-{k} accuracy: {topk_accuracy_value * 100:.2f}%')

        # Calculate and return top-k accuracy as a list of values
        topk_accuracy_values = [
            paddle.metric.accuracy(self.all_outputs, self.all_labels, k=k).item() if paddle.metric.accuracy(
                self.all_outputs, self.all_labels, k=k).ndim == 0 else
            paddle.metric.accuracy(self.all_outputs, self.all_labels, k=k).numpy()[0]
            for k in self.topk
        ]

        # Create the dictionary with the results
        results = {
            'f1_score': f1,
            'class_accuracies': [(class_correct[i] / class_total[i]) if class_total[i] > 0 else None for i in
                                 range(num_classes)],
            'topk_accuracy': topk_accuracy_values
        }



# 原来
    # https://github.com/PaddlePaddle/PaddleVideo/issues/591
    # def accumulate(self):
    #     """accumulate, compute, and show metrics when finished all iters.
    #     """
    #     self.all_outputs = paddle.concat(self.all_outputs, axis=0)
    #     self.all_labels = paddle.concat(self.all_labels, axis=0)
    #     topk_out, topk_indices = paddle.topk(self.all_outputs, k=1)
    #     # topk_out, topk_indices = paddle.topk(self.all_labels, k=1)
    #     # print(topk_out)
    #     # print(topk_indices)  # predict_label
    #     # print(self.all_labels)  # true_label
    #     print("topk_out:", topk_out)
    #     print("topk_indices:", topk_indices)  # predict_label
    #     print("self.all_labels:", self.all_labels)  # true_label
    #     predict_label = topk_indices.numpy()
    #     true_label = self.all_labels.numpy()
    #     predict_label = predict_label.flatten()
    #     true_label = true_label.flatten()
    #
    #     f1 = f1_score(true_label, predict_label, average='weighted')
    #     print("f1:", f1)
    #     result_str = []
    #     for _k in self.topk:
    #         topk_val = paddle.metric.accuracy(input=self.all_outputs,
    #                                           label=self.all_labels,
    #                                           k=_k).item()
    #         result_str.append(f"avg_acc{_k}={topk_val}")
    #     result_str = ", ".join(result_str)
    #     logger.info(f"[TEST] finished, {result_str}")
