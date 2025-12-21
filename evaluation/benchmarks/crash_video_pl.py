import json
import os
import re
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from .base import BaseVideoEvalDataset


class CrashPrecursorLocalizationDataset(BaseVideoEvalDataset):
    """
    Crash 视频前兆（precursor）定位任务 Dataset.

    - 适配 JSON 结构类似：
      [
        {
          "video": ["videos/dada_6_104.mp4"],
          "conversations": [
            {"from": "human", "value": "<video>..."},
            {"from": "gpt",   "value": "4.3 seconds"}
          ],
          "source": "DADA2000"
        },
        ...
      ]

    - data_root 预期为：例如 /home/kaidi/VideoLLaMA3/data
      其中包含：
        - crashchat_dada_video_precursor_localization_test.json
        - videos/dada_xxx.mp4
    """

    MODAL: str = "video"          # 必须是 "video"，evaluate.py 会用到
    BENCHMARK_TYPE: str = "precursor"  # 名字随便，后面不会调用基类的 mcqa 逻辑

    # ========= 0) 抽象方法：为兼容基类，返回该样本的指令文本 =========
    def generate_instruction(self, data_id: Union[int, str]) -> str:
        """
        用于兼容 BaseVideoEvalDataset / BaseEvalDataset 的抽象接口。

        这里就简单地返回我们在 load_data() 里存储的 question 字段。
        这样如果将来有通用的 prompt 构造逻辑要用 generate_instruction，
        也能拿到相同的文本。
        """
        if isinstance(data_id, str):
            data_id = int(data_id)

        meta = self.data_dict[data_id]
        return meta.get("question", "")

    # ========= 1) 加载数据 =========
    def load_data(self, data_root: str) -> Dict[int, Any]:
        """
        BaseVideoEvalDataset.__init__ 会调用这个函数，把它返回的 data_dict 存到 self.data_dict 里。

        这里先写死一个 precursor localization 的 JSON 文件名：
          crashchat_dada_video_precursor_localization_test.json

        后续你可以按需要改成根据 benchmark_name 区分不同 json 文件，
        或者在 build_dataset 里塞额外参数（比如 split 名）。
        """
        json_file = os.path.join(
            data_root,
            "crashchat_dada_video_precursor_localization_test.json",
        )
        with open(json_file, "r") as f:
            records: List[Dict[str, Any]] = json.load(f)

        data_dict: Dict[int, Any] = {}
        idx = 0

        for rec in records:
            # 1) 视频路径： "video": ["videos/dada_6_104.mp4"]
            video_list = rec.get("video", [])
            if not video_list:
                # 没有视频就跳过
                continue
            rel_video_path = video_list[0]
            video_path = os.path.join(data_root, rel_video_path)

            # 2) prompt / ground truth
            convs = rec.get("conversations", [])
            if len(convs) < 2:
                # 至少应该有人类问题 + gpt 答案
                continue

            human_prompt = convs[0].get("value", "")
            gt_str = convs[1].get("value", "")

            # 解析 GT："4.3 seconds" -> 4.3
            gt_time = self._parse_single_time(gt_str)

            meta_data = {
                # ====== required fields for data loading ======
                "video_path": video_path,
                "start_time": None,   # 这里不做 clip 截取，给 None 即可
                "end_time": None,

                # ====== required fields for evaluation / bookkeeping ======
                "task_type": "crash_precursor_localization",
                "ground_truth": gt_time,   # 单个时间点（float）

                # ====== custom fields for prompt 构造 ======
                "question": human_prompt,
                "source": rec.get("source", ""),
            }

            data_dict[idx] = meta_data
            idx += 1

        return data_dict

    # ========= 2) 把一条样本包装成模型需要的输入 =========
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        官方 evaluate.py 期望 __getitem__ 返回一个 dict：
          {
            "data_ids": [...],
            "image_inputs": {...},
            "text_inputs": [...],
          }

        后面会这样用：
          data_ids = data["data_ids"]
          text_inputs = data["text_inputs"]
          data_dict = {**data["image_inputs"], **text_input}
          response = mm_infer(...)
        """
        aggregated_data = self._aggregated_data_list[idx]

        # 1) 加载视频帧
        try:
            frames, timestamps = self.processor.load_video(
                aggregated_data["video_path"],
                start_time=aggregated_data["start_time"],
                end_time=aggregated_data["end_time"],
                precise_time=True,
                fps=self.fps,
                max_frames=self.max_frames,
            )
            image_inputs = self.processor.process_images(
                [frames],
                merge_size=2,
                return_tensors="pt",
            )
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"Failed to load video: {aggregated_data}")
            exit()

        # 2) 为这个视频上所有 data_id 构造 text_inputs（通常一个视频一个样本）
        text_inputs = []
        for data_id in aggregated_data["data_ids"]:
            meta_data = self.data_dict[data_id]
            question = meta_data["question"]

            # 构造 conversation：一个 video + 一个 text prompt
            content_list = [
                {
                    "type": "video",
                    "num_frames": len(frames),
                    "timestamps": timestamps,
                },
                {
                    "type": "text",
                    "text": question,
                },
            ]
            conversation = [
                {
                    "role": "user",
                    "content": content_list,
                }
            ]

            prompt = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

            text_inputs.append(
                self.processor.process_text(
                    prompt,
                    image_inputs,
                    padding=False,
                    padding_side=None,
                    return_tensors="pt",
                )
            )

        data = {
            "data_ids": aggregated_data["data_ids"],
            "image_inputs": image_inputs,
            "text_inputs": text_inputs,
        }
        return data

    # ========= 3) 解析模型输出 =========
    def process_response(self, data_id: Union[int, str], response: str) -> float:
        """
        把模型输出的字符串解析成一个时间点（秒数）.

        期望格式类似：
          "4.3 seconds"
          "time: 4.3s"
          "4.3s"
        """
        t = self._parse_single_time(response)
        return t

    # ========= 4) 评估：计算 MAE / 容差 accuracy =========
    def evaluate(
        self,
        results: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        results: 来自 evaluate.py 的列表：
          [
            {
              "data_id": data_id,
              "response": response_str,
              "prediction": pred_time,
            },
            ...
          ]
        返回：
          metrics: dict[str, float]
          infos:   List[Dict]（通常直接返回 results 或加上 GT 信息）
        """

        gt_times: List[float] = []
        pred_times: List[float] = []
        enhanced_infos: List[Dict[str, Any]] = []

        for item in results:
            data_id = item["data_id"]
            pred_time = item["prediction"]

            # 某些样本如果你以后改成返回 None，这里可以在此处跳过
            # 目前 _parse_single_time 永远返回 float，不会是 None
            if pred_time is None:
                continue

            gt_time = self.data_dict[data_id]["ground_truth"]

            gt_times.append(gt_time)
            pred_times.append(pred_time)

            info = {
                "data_id": data_id,
                "video_path": self.data_dict[data_id]["video_path"],
                "prompt": self.data_dict[data_id]["question"],
                "gt_time": gt_time,
                "pred_time": pred_time,
                "raw_response": item["response"],
            }
            enhanced_infos.append(info)

        if len(gt_times) == 0:
            # 没有有效样本，避免除以 0
            metrics = {
                "MAE": -1.0,
                "Acc_1s": 0.0,
                "Num_samples": 0,
            }
            return metrics, enhanced_infos

        gt_times_arr = np.array(gt_times, dtype=float)
        pred_times_arr = np.array(pred_times, dtype=float)

        mae = float(np.mean(np.abs(pred_times_arr - gt_times_arr)))

        # 例如：误差在 1 秒以内算命中
        tol = 1.0
        acc_mask = np.abs(pred_times_arr - gt_times_arr) <= tol
        acc_1s = float(np.mean(acc_mask.astype(np.float32)))

        metrics = {
            "MAE": mae,
            "Acc_1s": acc_1s,
            "Num_samples": float(len(gt_times_arr)),
        }
        return metrics, enhanced_infos

    # ========= 工具函数：解析单个时间点 =========
    @staticmethod
    def _parse_single_time(text: str) -> float:
        """
        从字符串中解析出一个时间点（秒），返回 float.

        支持例子：
          "4.3 seconds"
          "time: 4.3s"
          "4.3s"
          "t = 4.3"
        策略：
          - 找到所有数字串（含小数），取第一个作为时间
          - 解析失败则返回 0.0
        """
        if not isinstance(text, str):
            return 0.0

        nums = re.findall(r"[\d\.]+", text)
        if len(nums) >= 1:
            try:
                t = float(nums[0])
                return t
            except Exception:
                pass

        # 实在解析不了，就给个 0.0
        # 这里也可以选择 raise 或者打印 warning
        # print(f"[Warning] Failed to parse single time from: {text}")
        return 0.0
