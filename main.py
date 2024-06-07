# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定只使用第一张显卡进行训练
import argparse
import random

import numpy as np
import paddle

from paddlevideo.tasks import (test_model, train_dali, train_model,
                               train_model_multigrid)
from paddlevideo.utils import get_config, get_dist_info


def parse_args():
    parser = argparse.ArgumentParser("PaddleVideo train script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        # default='configs/recognition/pptsm/pptsm_task_VideoClassfy_frames_dense.yaml',
                        # default='configs/recognition/pptsm/pptsm_task_VideoClassfy_frames_dense_modify.yaml',
                        # default='configs/recognition/pptsm/pptsm_task_VideoClassfy_frames_dense_modify_ucf24.yaml',

                        default='configs/recognition/pptsm/pptsm_task_VideoClassfy_frames_dense_ucf101.yaml',
                        # default='configs/recognition/pptsm/pptsm_task_VideoClassfy_frames_dense_modify_ucf101.yaml',

                        # default='configs/recognition/pptsm/pptsm_task_VideoClassfy_frames_dense_hmdb51.yaml',
                        # default='configs/recognition/pptsm/pptsm_task_VideoClassfy_frames_dense_modify_hmdb51.yaml',
                        # default='configs/recognition/pptsm/pptsm_task_VideoClassfy_frames_dense_sthv1.yaml',
                        # default='configs/recognition/pptsm/pptsm_task_VideoClassfy_frames_dense_modify_sthv1.yaml',
                        help='config file path')
    # parser.add_argument('-o',
    #                     '--override',
    #                     action='append',
    #                     default=[],
    #                     help='config options to be overridden')
    parser.add_argument('-o',
                        '--override',
                        action='append',
                        default=['resume_epoch=44'],
                        help='config options to be overridden')
    parser.add_argument('--test',
                        action='store_true',
                        help='whether to test a model')
    parser.add_argument('--train_dali',
                        action='store_true',
                        help='whether to use dali to speed up training')
    parser.add_argument('--multigrid',
                        action='store_true',
                        help='whether to use multigrid training')
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        help='weights for finetuning or testing')
    parser.add_argument('--fleet',
                        action='store_true',
                        help='whether to use fleet run distributed training')
    parser.add_argument('--amp',
                        action='store_true',
                        help='whether to open amp training.')
    parser.add_argument(
        '--amp_level',
        type=str,
        default=None,
        help="optimize level when open amp training, can only be 'O1' or 'O2'.")
    parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--seed',
        type=int,
        default=1234,
        help='fixed all random seeds when the program is running')
    parser.add_argument(
        '--max_iters',
        type=int,
        default=None,
        help='max iterations when training(this arg only used in test_tipc)')
    parser.add_argument('--log_dir',
                        type=str,
                        default='log/modify_ucf24',
                        help='log directory path')
    parser.add_argument(
        '-p',
        '--profiler_options',
        type=str,
        default=None,
        help='The option of profiler, which should be in format '
        '\"key1=value1;key2=value2;key3=value3\".')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = get_config(args.config, overrides=args.override)

    # enable to use npu if paddle is built with npu
    if paddle.is_compiled_with_custom_device('npu') :
    # if paddle.device.is_compiled_with_xpu():
        cfg.__setattr__("use_npu", True)
    elif paddle.device.is_compiled_with_xpu():
        cfg.__setattr__("use_xpu", True)

    # set seed if specified
    seed = args.seed
    if seed is not None:
        assert isinstance(
            seed, int), f"seed must be a integer when specified, but got {seed}"
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    # set amp_level if amp is enabled
    if args.amp:
        if args.amp_level is None:
            args.amp_level = 'O1'  # set defaualt amp_level to 'O1'
        else:
            assert args.amp_level in [
                'O1', 'O2'
            ], f"amp_level must be 'O1' or 'O2' when amp enabled, but got {args.amp_level}."

    _, world_size = get_dist_info()
    # print("_:", _)
    # print("world_size:", world_size)
    parallel = world_size != 1
    # parallel = True
    if parallel:
        print("parallel == true")
        paddle.distributed.init_parallel_env()
    # if paddle.distributed.get_rank() == 0:
    #     print("分布式环境初始化成功。")
    # print(f"当前进程的全局rank: {paddle.distributed.get_rank()}, 全局world size: {paddle.distributed.get_world_size()}")

    if args.test:
        test_model(cfg, weights=args.weights, parallel=parallel)
    elif args.train_dali:
        train_dali(cfg, weights=args.weights, parallel=parallel)
    elif args.multigrid:
        train_model_multigrid(cfg,
                              world_size=world_size,
                              validate=args.validate)
    else:
        train_model(cfg,
                    weights=args.weights,
                    parallel=parallel,
                    validate=args.validate,
                    use_fleet=args.fleet,
                    use_amp=args.amp,
                    amp_level=args.amp_level,
                    max_iters=args.max_iters,
                    profiler_options=args.profiler_options)


if __name__ == '__main__':
    main()
