import json
import os
import re
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from .base import BaseVideoEvalDataset


class CrashAnticipationLocalizationDataset(BaseVideoEvalDataset):
    """
    Crash 视频事故预见 (anticipation localization) 评估 Dataset.

    - 适配 JSON 结构类似：
      [
        {
          "video": ["videos/cap_21_008073.mp4"],
          "conversations": [
            {
              "from": "human",
              "value": "<video>... 输出格式: 'timestamp seconds' ..."
            },
            {
              "from": "gpt",
              "value": "1.0 seconds"
            }
          ],
          "source": "MM-AU-CAP"
        },
        ...
      ]

    - data_root 预期为：例如 /home/kaidi/VideoLLaMA3/data
      其中包含：
        - crashchat_dada_video_total_anticipation_localization_test.json
        - videos/xxx.mp4
    """

    MODAL: str = "video"      # 必须是 "video"，evaluate.py 会用到
    BENCHMARK_TYPE: str = "anticipation_localization"  # 任意区分名称

    # ========= 0) 抽象方法：为兼容基类，返回该样本的指令文本 =========
    def generate_instruction(self, data_id: Union[int, str]) -> str:
        """
        用于兼容 BaseVideoEvalDataset / BaseEvalDataset 的抽象接口。

        这里就简单地返回我们在 load_data() 里存储的 question 字段。
        """
        if isinstance(data_id, str):
            data_id = int(data_id)

        meta = self.data_dict[data_id]
        return meta.get("question", "")

    # ========= 1) 加载数据 =========
    def load_data(self, data_root: str) -> Dict[int, Any]:
        """
        BaseVideoEvalDataset.__init__ 会调用这个函数，把它返回的 data_dict 存到 self.data_dict 里。

        这里先写死一个 anticipation localization 的 JSON 文件名：
          crashchat_dada_video_total_anticipation_localization_test.json
        """
        json_file = os.path.join(
            data_root,
            "crashchat_dada_video_total_anticipation_localization_test.json",
        )
        with open(json_file, "r") as f:
            records: List[Dict[str, Any]] = json.load(f)

        data_dict: Dict[int, Any] = {}
        idx = 0

        for rec in records:
            # 1) 视频路径："video": ["videos/cap_21_008073.mp4"]
            video_list = rec.get("video", [])
            if not video_list:
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

            # 解析 GT："1.0 seconds" -> 1.0
            gt_ts = self._parse_timestamp(gt_str)
            if gt_ts is None:
                # 解析失败就跳过该样本
                # 也可以改成 continue / raise，看你需求
                continue

            meta_data = {
                # ====== required fields for data loading ======
                "video_path": video_path,
                "start_time": None,
                "end_time": None,

                # ====== required fields for evaluation / bookkeeping ======
                "task_type": "crash_anticipation_localization",
                "ground_truth": gt_ts,   # float

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
    def process_response(self, data_id: Union[int, str], response: str) -> Union[float, None]:
        """
        把模型输出的字符串解析成单一时间戳（秒）.

        期望格式类似：
          "3.4 seconds"
          "3.4s"
          "timestamp: 3.4"
        """
        ts = self._parse_timestamp(response)
        return ts

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
              "prediction": pred_ts,  # float 或 None
            },
            ...
          ]
        返回：
          metrics: dict[str, float]
          infos:   List[Dict]（包含 GT + 预测 + 原始 response）
        """

        gt_ts_list: List[float] = []
        pred_ts_list: List[float] = []
        enhanced_infos: List[Dict[str, Any]] = []

        for item in results:
            data_id = item["data_id"]
            pred_ts = item["prediction"]

            # 某些样本可能解析失败（prediction=None）
            if pred_ts is None:
                continue

            gt_ts = self.data_dict[data_id]["ground_truth"]

            gt_ts_list.append(gt_ts)
            pred_ts_list.append(pred_ts)

            info = {
                "data_id": data_id,
                "video_path": self.data_dict[data_id]["video_path"],
                "prompt": self.data_dict[data_id]["question"],
                "gt_time": gt_ts,
                "pred_time": pred_ts,
                "raw_response": item["response"],
            }
            enhanced_infos.append(info)

        if len(gt_ts_list) == 0:
            metrics = {
                "MAE": -1.0,
                "Acc_1s": 0.0,
                "Num_samples": 0.0,
            }
            return metrics, enhanced_infos

        gt_ts_arr = np.array(gt_ts_list, dtype=float)
        pred_ts_arr = np.array(pred_ts_list, dtype=float)

        mae = float(np.mean(np.abs(pred_ts_arr - gt_ts_arr)))

        tol = 1.0  # 容差 1 秒
        acc_mask = np.abs(pred_ts_arr - gt_ts_arr) <= tol
        acc_1s = float(np.mean(acc_mask.astype(np.float32)))

        metrics = {
            "MAE": mae,
            "Acc_1s": acc_1s,
            "Num_samples": float(len(gt_ts_list)),
        }
        return metrics, enhanced_infos

    # ========= 工具函数：解析单一时间戳 =========
    @staticmethod
    def _parse_timestamp(text: Any) -> Union[float, None]:
        """
        从字符串中解析出一个时间戳（秒），返回 float.

        支持例子：
          "1.0 seconds"
          "4.5 s"
          "timestamp: 3.4"
        如果解析失败，返回 None。
        """
        if not isinstance(text, str):
            return None

        # 找到第一个数字
        m = re.search(r"([\d\.]+)", text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None

        return None
