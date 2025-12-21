import json
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from .base import BaseVideoEvalDataset


class CrashAccidentRecognitionDataset(BaseVideoEvalDataset):
    """
    Crash 视频事故识别（Yes/No）评估 Dataset

    适配 JSON 结构类似：
      [
        {
          "video": ["videos/dada_38_016.mp4"],
          "conversations": [
            {
              "from": "human",
              "value": "<video>\\nYou are a traffic accident engineer. ..."
            },
            {
              "from": "gpt",
              "value": "Yes"
            }
          ],
          "source": "DADA2000"
        },
        ...
      ]

    data_root 预期为：例如 /home/kaidi/VideoLLaMA3/data
      其中包含：
        - crashchat_dada_video_accident_recognition_test.json
        - videos/dada_xxx.mp4
    """

    MODAL: str = "video"  # 必须是 "video"，evaluate.py 会用到
    BENCHMARK_TYPE: str = "accident_recognition"  # 任意非 mcqa 名字即可

    # ========= 0) 抽象方法：为兼容基类，返回该样本的指令文本 =========
    def generate_instruction(self, data_id: Union[int, str]) -> str:
        """
        用于兼容 BaseVideoEvalDataset / BaseEvalDataset 的抽象接口。

        这里直接返回在 load_data() 里存的 question 字段。
        """
        if isinstance(data_id, str):
            data_id = int(data_id)
        meta = self.data_dict[data_id]
        return meta.get("question", "")

    # ========= 1) 加载数据 =========
    def load_data(self, data_root: str) -> Dict[int, Any]:
        """
        这里先写死一个 accident recognition 的 JSON 文件名：
          crashchat_dada_video_accident_recognition_test.json

        如果你想改成 train/val/test，可之后再做参数化。
        """
        json_file = os.path.join(
            data_root,
            "crashchat_dada_video_total_accident_recognition_test.json",
        )
        with open(json_file, "r") as f:
            records: List[Dict[str, Any]] = json.load(f)

        data_dict: Dict[int, Any] = {}
        idx = 0

        for rec in records:
            # 1) 视频路径："video": ["videos/dada_38_016.mp4"]
            video_list = rec.get("video", [])
            if not video_list:
                continue
            rel_video_path = video_list[0]
            video_path = os.path.join(data_root, rel_video_path)

            # 2) prompt / ground truth
            convs = rec.get("conversations", [])
            if len(convs) < 2:
                # 至少需要 human + gpt
                continue

            human_prompt = convs[0].get("value", "")
            gt_str = convs[1].get("value", "")

            # 解析 GT: "Yes"/"No" -> 1/0
            gt_label = self._parse_yes_no(gt_str)

            meta_data = {
                # ====== required fields for data loading ======
                "video_path": video_path,
                "start_time": None,
                "end_time": None,

                # ====== required fields for evaluation / bookkeeping ======
                "task_type": "crash_accident_recognition",
                "ground_truth": gt_label,  # int: 1(Yes) / 0(No)

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
    def process_response(self, data_id: Union[int, str], response: str) -> Union[int, None]:
        """
        把模型输出字符串解析成 1/0.

        期望模型输出尽量是：
          "Yes" 或 "No"
        也做了一些健壮处理：
          - 如果包含 "yes"（忽略大小写），则视为 1
          - 如果包含 "no"（忽略大小写），则视为 0
          - 若无法判断，返回 None
        """
        label = self._parse_yes_no(response)
        # 返回 None 表示解析失败，在 evaluate() 里会跳过
        return label if label in (0, 1) else None

    # ========= 4) 评估：计算 Accuracy / Precision / Recall / F1 =========
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
              "prediction": pred_label,  # 1/0 或 None
            },
            ...
          ]
        返回：
          metrics: dict[str, float]
          infos:   List[Dict]（通常直接返回 results 或加上 GT 信息）
        """
        y_true: List[int] = []
        y_pred: List[int] = []
        enhanced_infos: List[Dict[str, Any]] = []

        for item in results:
            data_id = item["data_id"]
            pred = item["prediction"]

            # 某些样本可能解析失败（prediction=None）
            if pred is None:
                continue

            gt = self.data_dict[data_id]["ground_truth"]
            y_true.append(gt)
            y_pred.append(pred)

            info = {
                "data_id": data_id,
                "video_path": self.data_dict[data_id]["video_path"],
                "prompt": self.data_dict[data_id]["question"],
                "gt_label": "Yes" if gt == 1 else "No",
                "pred_label": "Yes" if pred == 1 else "No",
                "raw_response": item["response"],
            }
            enhanced_infos.append(info)

        if len(y_true) == 0:
            metrics = {
                "Accuracy": 0.0,
                "Precision_Yes": 0.0,
                "Recall_Yes": 0.0,
                "F1_Yes": 0.0,
                "Num_samples": 0.0,
                "TP": 0.0,
                "TN": 0.0,
                "FP": 0.0,
                "FN": 0.0,
            }
            return metrics, enhanced_infos

        y_true_arr = np.array(y_true, dtype=int)
        y_pred_arr = np.array(y_pred, dtype=int)

        # confusion matrix elements for "Yes" class (1)
        tp = float(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
        tn = float(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
        fp = float(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
        fn = float(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))

        n = float(len(y_true_arr))
        acc = (tp + tn) / n

        precision_yes = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_yes = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision_yes + recall_yes > 0:
            f1_yes = 2 * precision_yes * recall_yes / (precision_yes + recall_yes)
        else:
            f1_yes = 0.0

        metrics = {
            "Accuracy": float(acc),
            "Precision_Yes": float(precision_yes),
            "Recall_Yes": float(recall_yes),
            "F1_Yes": float(f1_yes),
            "Num_samples": float(n),
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        }
        return metrics, enhanced_infos

    # ========= 工具函数：解析 Yes / No =========
    @staticmethod
    def _parse_yes_no(text: Any) -> Union[int, None]:
        """
        解析字符串为 1/0：

        返回值：
          1 -> Yes / 有事故
          0 -> No  / 无事故
          None -> 无法解析
        """
        if not isinstance(text, str):
            return None

        s = text.strip().lower()

        # 如果整句就是 "yes"/"no"
        if s == "yes":
            return 1
        if s == "no":
            return 0

        # 宽松一些：包含 yes/no 的情况
        if "yes" in s and "no" not in s:
            return 1
        if "no" in s and "yes" not in s:
            return 0

        # 实在不行就返回 None
        return None
