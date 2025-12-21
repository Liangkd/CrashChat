import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from .base import BaseVideoEvalDataset


class CrashAccidentMeasureDataset(BaseVideoEvalDataset):
    """
    Crash 视频事故预防措施（Accident Measure, AM）评估 Dataset

    适配 JSON 结构类似：
      [
        {
          "video": ["videos/cap_21_008073.mp4"],
          "conversations": [
            {
              "from": "human",
              "value": "<video>\\nYou are a traffic accident analyst. Watch the video carefully ..."
            },
            {
              "from": "gpt",
              "value": "The vehicle drivers should not answer or make calls ..."
            }
          ],
          "source": "MM-AU-CAP"
        },
        ...
      ]

    data_root 预期为：例如 /lustre/nvwulf/scratch/kaidliang/VideoLLaMA3/data
      其中包含：
        - crashchat_dada_video_total_accident_measure_test.json
        - videos/xxx.mp4
    """

    MODAL: str = "video"              # 必须是 "video"，evaluate.py 会用到
    BENCHMARK_TYPE: str = "accident_measure"  # 用于区分任务类型

    # ========= 0) 抽象方法：为兼容基类，返回该样本的指令文本 =========
    def generate_instruction(self, data_id: Union[int, str]) -> str:
        """
        用于兼容 BaseVideoEvalDataset / BaseEvalDataset 的抽象接口。

        这里直接返回在 load_data() 里存储的 question 字段。
        """
        if isinstance(data_id, str):
            data_id = int(data_id)

        meta = self.data_dict[data_id]
        return meta.get("question", "")

    # ========= 1) 加载数据 =========
    def load_data(self, data_root: str) -> Dict[int, Any]:
        """
        BaseVideoEvalDataset.__init__ 会调用这个函数，把它返回的 data_dict 存到 self.data_dict 里。

        这里对应 AM 任务的 JSON：
          crashchat_dada_video_total_accident_measure_test.json
        """
        json_file = os.path.join(
            data_root,
            "crashchat_dada_video_total_accident_measure_test.json",
        )
        with open(json_file, "r") as f:
            records: List[Dict[str, Any]] = json.load(f)

        data_dict: Dict[int, Any] = {}
        idx = 0

        for rec in records:
            # 1) 视频路径："video": ["videos/cap_21_008073.mp4"]
            video_list = rec.get("video", [])
            if not video_list:
                # 没有视频就跳过
                continue
            rel_video_path = video_list[0]
            video_path = os.path.join(data_root, rel_video_path)

            # 2) prompt / ground truth prevention measures
            convs = rec.get("conversations", [])
            if len(convs) < 2:
                # 至少应该有人类问题 + gpt 答案
                continue

            human_prompt = convs[0].get("value", "")
            gt_text = convs[1].get("value", "")

            meta_data = {
                # ====== required fields for data loading ======
                "video_path": video_path,
                "start_time": None,   # 不截 clip，直接用全视频
                "end_time": None,

                # ====== required fields for evaluation / bookkeeping ======
                "task_type": "crash_accident_measure",
                "ground_truth": gt_text,  # str

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
    def process_response(self, data_id: Union[int, str], response: str) -> Union[str, None]:
        """
        把模型输出的字符串解析成预防措施文本。

        这里不做复杂解析，只做 strip：
          - 如果 response 不是字符串，返回 None
          - 否则返回去掉首尾空格后的字符串
        """
        if not isinstance(response, str):
            return None
        return response.strip()

    # ========= 3.5) 一个简单的 token F1 计算工具 =========
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # 最简单粗暴的分词：只保留字母数字，转小写
        tokens = re.findall(r"\w+", text.lower())
        return tokens

    @staticmethod
    def _f1_score(pred_tokens: List[str], gt_tokens: List[str]) -> float:
        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0

        pred_counter = Counter(pred_tokens)
        gt_counter = Counter(gt_tokens)

        # 交集 token 数
        overlap = sum((pred_counter & gt_counter).values())
        if overlap == 0:
            return 0.0

        precision = overlap / len(pred_tokens)
        recall = overlap / len(gt_tokens)
        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return float(f1)

    # ========= 4) 评估：用 token-level F1 作为一个简单的文本相似度指标 =========
    def evaluate(
        self,
        results: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        results: 来自 evaluate.py 的列表：
          [
            {
              "data_id": data_id,
              "response": response_str,    # 原始生成
              "prediction": pred_text,     # 经过 process_response 的文本（str 或 None）
            },
            ...
          ]

        返回：
          metrics: dict[str, float]
          infos:   List[Dict]（包含 GT + 预测 + 原始 response）
        """

        gts: List[str] = []
        preds: List[str] = []
        enhanced_infos: List[Dict[str, Any]] = []

        for item in results:
            data_id = item["data_id"]
            pred_text = item["prediction"]

            # 某些样本可能解析失败（prediction=None）
            if pred_text is None:
                continue

            gt_text = self.data_dict[data_id]["ground_truth"]

            gts.append(gt_text)
            preds.append(pred_text)

            info = {
                "data_id": data_id,
                "video_path": self.data_dict[data_id]["video_path"],
                "prompt": self.data_dict[data_id]["question"],
                "gt_text": gt_text,
                "pred_text": pred_text,
                "raw_response": item["response"],
            }
            enhanced_infos.append(info)

        if len(gts) == 0:
            metrics = {
                "Token_F1": 0.0,
                "Num_samples": 0.0,
            }
            return metrics, enhanced_infos

        # 计算平均 token F1
        f1_list: List[float] = []
        for gt, pred in zip(gts, preds):
            gt_tokens = self._tokenize(gt)
            pred_tokens = self._tokenize(pred)
            f1 = self._f1_score(pred_tokens, gt_tokens)
            f1_list.append(f1)

        avg_f1 = float(np.mean(f1_list))

        metrics = {
            "Token_F1": avg_f1,
            "Num_samples": float(len(gts)),
        }
        return metrics, enhanced_infos
