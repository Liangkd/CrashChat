import json
import os
import re
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from .base import BaseVideoEvalDataset


class CrashTemporalLocalizationDataset(BaseVideoEvalDataset):
    """
    通用的 Crash 视频任务 Dataset（当前先实现 temporal localization 版本）.

    - 适配 JSON 结构类似：
      [
        {
          "video": ["videos/dada_6_104.mp4"],
          "conversations": [
            {"from": "human", "value": "<video>..."},
            {"from": "gpt",   "value": "6.0 - 7.0 seconds"}
          ],
          "source": "DADA2000"
        },
        ...
      ]

    - data_root 预期为：例如 /home/kaidi/VideoLLaMA3/data
      其中包含：
        - crashchat_dada_video_temporal_localization_test.json
        - videos/dada_xxx.mp4
    """

    MODAL: str = "video"      # 必须是 "video"，evaluate.py 会用到
    BENCHMARK_TYPE: str = "temporal"  # 名字随便，后面不会调用基类的 mcqa 逻辑

    # ========= 0) 抽象方法：为兼容基类，返回该样本的指令文本 =========
    def generate_instruction(self, data_id: Union[int, str]) -> str:
        """
        用于兼容 BaseVideoEvalDataset / BaseEvalDataset 的抽象接口。

        这里就简单地返回我们在 load_data() 里存储的 question 字段。
        这样如果将来有通用的 prompt 构造逻辑要用 generate_instruction，
        也能拿到相同的文本。
        """
        # data_id 可能是 str，这里统一转成 int
        if isinstance(data_id, str):
            data_id = int(data_id)

        meta = self.data_dict[data_id]
        return meta.get("question", "")

    # ========= 1) 加载数据 ========= 
    def load_data(self, data_root: str) -> Dict[int, Any]:
        """
        BaseVideoEvalDataset.__init__ 会调用这个函数，把它返回的 data_dict 存到 self.data_dict 里。

        这里我们先写死一个 TL 的 JSON 文件名：
          crashchat_dada_video_temporal_localization_test.json

        后续你可以按需要改成根据 benchmark_name 区分不同 json 文件，
        或者在 build_dataset 里塞额外参数（比如 split 名）。
        """
        json_file = os.path.join(
            data_root,
            "crashchat_dada_video_total_temporal_localization_test.json",
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

            # 解析 GT："6.0 - 7.0 seconds" -> (6.0, 7.0)
            gt_start, gt_end = self._parse_time_range(gt_str)

            meta_data = {
                # ====== required fields for data loading ======
                "video_path": video_path,
                "start_time": None,   # 这里不做 clip 截取，给 None 即可
                "end_time": None,

                # ====== required fields for evaluation / bookkeeping ======
                "task_type": "crash_temporal_localization",
                "ground_truth": (gt_start, gt_end),

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
    def process_response(self, data_id: Union[int, str], response: str) -> Tuple[float, float]:
        """
        把模型输出的字符串解析成 (start, end) 秒数.

        期望格式类似：
          "3.4 - 5.2 seconds"
          "3.4-5.2s"
        """
        start, end = self._parse_time_range(response)
        return (start, end)

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
              "prediction": (pred_start, pred_end),
            },
            ...
          ]
        返回：
          metrics: dict[str, float]
          infos:   List[Dict]（通常直接返回 results 或加上 GT 信息）
        """

        gt_starts, gt_ends = [], []
        pred_starts, pred_ends = [], []
        enhanced_infos: List[Dict[str, Any]] = []

        for item in results:
            data_id = item["data_id"]
            pred = item["prediction"]

            # 某些样本可能解析失败（我们约定 prediction=None）
            if pred is None:
                continue

            pred_start, pred_end = pred

            gt_start, gt_end = self.data_dict[data_id]["ground_truth"]

            gt_starts.append(gt_start)
            gt_ends.append(gt_end)
            pred_starts.append(pred_start)
            pred_ends.append(pred_end)

            info = {
                "data_id": data_id,
                "video_path": self.data_dict[data_id]["video_path"],
                "prompt": self.data_dict[data_id]["question"],
                "gt_start": gt_start,
                "gt_end": gt_end,
                "pred_start": pred_start,
                "pred_end": pred_end,
                "raw_response": item["response"],
            }
            enhanced_infos.append(info)

        if len(gt_starts) == 0:
            # 没有有效样本，避免除以 0
            metrics = {
                "MAE_start": -1.0,
                "MAE_end": -1.0,
                "MAE_mean": -1.0,
                "Acc_1s": 0.0,
                "Num_samples": 0,
            }
            return metrics, enhanced_infos

        gt_starts = np.array(gt_starts, dtype=float)
        gt_ends = np.array(gt_ends, dtype=float)
        pred_starts = np.array(pred_starts, dtype=float)
        pred_ends = np.array(pred_ends, dtype=float)

        mae_start = float(np.mean(np.abs(pred_starts - gt_starts)))
        mae_end = float(np.mean(np.abs(pred_ends - gt_ends)))
        mae_mean = float((mae_start + mae_end) / 2.0)

        # 例如：同时满足起止都在 1 秒以内，算命中
        tol = 1.0
        acc_mask = (
            (np.abs(pred_starts - gt_starts) <= tol)
            & (np.abs(pred_ends - gt_ends) <= tol)
        )
        acc_1s = float(np.mean(acc_mask.astype(np.float32)))

        metrics = {
            "MAE_start": mae_start,
            "MAE_end": mae_end,
            "MAE_mean": mae_mean,
            "Acc_1s": acc_1s,
            "Num_samples": float(len(gt_starts)),
        }
        return metrics, enhanced_infos

    # ========= 工具函数：解析时间区间 =========
    @staticmethod
    def _parse_time_range(text: str) -> Tuple[float, float]:
        """
        从字符串中解析出 "start - end" 的数字，返回 (start, end).

        支持例子：
          "6.0 - 7.0 seconds"
          "6.0-7.0s"
          "start: 6.0, end: 7.0"
        如果解析失败，就返回 (0.0, 0.0)，并且你也可以根据需要打印 warning。
        """
        if not isinstance(text, str):
            return 0.0, 0.0

        # 最常见情况： "6.0 - 7.0"
        m = re.search(r"([\d\.]+)\s*-\s*([\d\.]+)", text)
        if m:
            try:
                s = float(m.group(1))
                e = float(m.group(2))
                return s, e
            except Exception:
                pass

        # 其它简单 fallback，比如 "start: 6.0, end: 7.0"
        nums = re.findall(r"[\d\.]+", text)
        if len(nums) >= 2:
            try:
                s = float(nums[0])
                e = float(nums[1])
                return s, e
            except Exception:
                pass

        # 实在解析不了，就给个 (0,0)
        # 这里也可以选择 raise 或者打印 warning
        # print(f"[Warning] Failed to parse time range from: {text}")
        return 0.0, 0.0
