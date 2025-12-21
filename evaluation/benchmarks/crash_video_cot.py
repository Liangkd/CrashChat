import json
import os
import re
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np

from .base import BaseVideoEvalDataset


class CrashCoTDataset(BaseVideoEvalDataset):
    """
    Crash chain-of-thought 视频评估 Dataset.

    适配 JSON 结构类似：
      [
        {
          "video": ["videos/dada_6_104.mp4"],
          "conversations": [
            { "from": "human", "value": "<video>... Step 1 ..." },
            { "from": "gpt",   "value": "Yes" },

            { "from": "human", "value": "Step 2: ..." },
            { "from": "gpt",   "value": "ego-car hits a motorbike" },

            { "from": "human", "value": "Step 3: ..." },
            { "from": "gpt",   "value": "6.0 - 7.0 seconds" },

            { "from": "human", "value": "Step 4: ..." },
            { "from": "gpt",   "value": "4.3 seconds" }
          ],
          "source": "DADA2000"
        },
        ...
      ]

    评估目标（per-sample）：
      - Step 1: Yes/No 准确率
      - Step 3: Crash temporal localization（start/end）MAE + 1s 容差准确率
      - Step 4: Precursor timestamp MAE + 1s 容差准确率

    注意：
      - 如果 GT 为 "N/A"，对应 Step 3/4 不参与 MAE/Acc 计算
      - 模型输出可以是一整段包含多步答案的文本，process_response 会尝试解析其中的
        Step 1 / Step 3 / Step 4 信息。
    """

    MODAL: str = "video"
    BENCHMARK_TYPE: str = "cot"

    # ========= 0) 抽象方法：为兼容基类，返回该样本的指令文本 =========
    def generate_instruction(self, data_id: Union[int, str]) -> str:
        if isinstance(data_id, str):
            data_id = int(data_id)
        meta = self.data_dict[data_id]
        return meta.get("question", "")

    # ========= 1) 加载数据 =========
    def load_data(self, data_root: str) -> Dict[int, Any]:
        """
        这里先写死使用：
          crashchat_dada_video_test_cot.json
        作为 CoT 评估用的 test 集。
        """
        json_file = os.path.join(
            data_root,
            "crashchat_dada_video_test_cot.json",
        )
        with open(json_file, "r") as f:
            records: List[Dict[str, Any]] = json.load(f)

        data_dict: Dict[int, Any] = {}
        idx = 0

        for rec in records:
            # 1) 视频路径
            video_list = rec.get("video", [])
            if not video_list:
                continue
            rel_video_path = video_list[0]
            video_path = os.path.join(data_root, rel_video_path)

            # 2) conversations: human/gpt 交替
            convs = rec.get("conversations", [])
            if len(convs) < 8:
                # 至少要有 4 个 human + 4 个 gpt
                continue

            # --- human prompt：把所有 human value 拼成一个 CoT prompt ---
            human_parts = [c["value"] for c in convs if c.get("from") == "human"]
            if not human_parts:
                continue
            full_human_prompt = "\n\n".join(human_parts)

            # --- Ground Truth from conversations ---
            # 约定：
            #   0: human (Step 1)
            #   1: gpt   (Step 1 answer)
            #   2: human (Step 2)
            #   3: gpt   (Step 2 answer)
            #   4: human (Step 3)
            #   5: gpt   (Step 3 answer: time range)
            #   6: human (Step 4)
            #   7: gpt   (Step 4 answer: precursor timestamp)
            gt_step1_raw = convs[1].get("value", "").strip()
            gt_step2_raw = convs[3].get("value", "").strip()
            gt_step3_raw = convs[5].get("value", "").strip()
            gt_step4_raw = convs[7].get("value", "").strip()

            gt_step1 = self._normalize_yes_no(gt_step1_raw)
            gt_step2 = gt_step2_raw

            # Step 3: crash temporal range
            gt_tl_start, gt_tl_end = self._parse_time_range_allow_na(gt_step3_raw)

            # Step 4: precursor timestamp
            gt_precursor = self._parse_timestamp_allow_na(gt_step4_raw)

            meta_data = {
                # ====== required fields for data loading ======
                "video_path": video_path,
                "start_time": None,
                "end_time": None,

                # ====== ground truth for evaluation ======
                "gt_step1": gt_step1,                 # "yes"/"no" 或 None
                "gt_step2": gt_step2,                 # caption 文本（不用于严格 metric，可供人工分析）
                "gt_tl_start": gt_tl_start,           # float or None
                "gt_tl_end": gt_tl_end,               # float or None
                "gt_precursor": gt_precursor,         # float or None

                # ====== custom fields for prompt 构造 ======
                "question": full_human_prompt,
                "source": rec.get("source", ""),
                "task_type": "crash_cot",
            }

            data_dict[idx] = meta_data
            idx += 1

        return data_dict

    # ========= 2) 把一条样本包装成模型需要的输入 =========
    def __getitem__(self, idx: int) -> Dict[str, Any]:
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

        # 2) 为这个视频上所有 data_id 构造 text_inputs（一般 1 个样本）
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
    def process_response(
        self,
        data_id: Union[int, str],
        response: str,
    ) -> Dict[str, Any]:
        """
        解析模型对 CoT prompt 的完整输出，抽取：

          - step1: "yes"/"no" 或 None
          - tl:    (start, end) or (None, None)
          - precursor: float or None

        （Step 2 caption 直接保留原文，主要用于 enhanced_infos 中人工对比）
        """
        if not isinstance(response, str):
            return {
                "step1": None,
                "tl": (None, None),
                "precursor": None,
                "step2": None,
            }

        step1_pred = self._parse_step1_from_response(response)
        step2_pred = self._parse_step2_from_response(response)
        tl_start_pred, tl_end_pred = self._parse_step3_from_response(response)
        precursor_pred = self._parse_step4_from_response(response)

        return {
            "step1": step1_pred,
            "tl": (tl_start_pred, tl_end_pred),
            "precursor": precursor_pred,
            "step2": step2_pred,
        }

    # ========= 4) 评估 =========
    def evaluate(
        self,
        results: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:

        # Step1: yes/no
        s1_gts: List[int] = []
        s1_preds: List[int] = []

        # Step3: crash temporal range
        tl_gt_starts: List[float] = []
        tl_gt_ends: List[float] = []
        tl_pred_starts: List[float] = []
        tl_pred_ends: List[float] = []

        # Step4: precursor
        pre_gts: List[float] = []
        pre_preds: List[float] = []

        enhanced_infos: List[Dict[str, Any]] = []

        for item in results:
            data_id = item["data_id"]
            pred = item["prediction"]  # dict from process_response
            raw_response = item["response"]

            meta = self.data_dict[data_id]

            gt_step1 = meta["gt_step1"]
            gt_step2 = meta["gt_step2"]
            gt_tl_start = meta["gt_tl_start"]
            gt_tl_end = meta["gt_tl_end"]
            gt_precursor = meta["gt_precursor"]

            if not isinstance(pred, dict):
                # 解析失败
                pred_step1 = None
                (pred_tl_start, pred_tl_end) = (None, None)
                pred_precursor = None
                pred_step2 = None
            else:
                pred_step1 = pred.get("step1", None)
                (pred_tl_start, pred_tl_end) = pred.get("tl", (None, None))
                pred_precursor = pred.get("precursor", None)
                pred_step2 = pred.get("step2", None)

            # ======= Step1 Acc =======
            if gt_step1 is not None and pred_step1 is not None:
                s1_gts.append(1 if gt_step1 == "yes" else 0)
                s1_preds.append(1 if pred_step1 == "yes" else 0)

            # ======= Step3 TL metrics =======
            if (
                gt_tl_start is not None
                and gt_tl_end is not None
                and pred_tl_start is not None
                and pred_tl_end is not None
            ):
                tl_gt_starts.append(gt_tl_start)
                tl_gt_ends.append(gt_tl_end)
                tl_pred_starts.append(pred_tl_start)
                tl_pred_ends.append(pred_tl_end)

            # ======= Step4 precursor metrics =======
            if gt_precursor is not None and pred_precursor is not None:
                pre_gts.append(gt_precursor)
                pre_preds.append(pred_precursor)

            info = {
                "data_id": data_id,
                "video_path": meta["video_path"],
                "prompt": meta["question"],
                "gt_step1": gt_step1,
                "pred_step1": pred_step1,
                "gt_step2": gt_step2,
                "pred_step2": pred_step2,
                "gt_tl_start": gt_tl_start,
                "gt_tl_end": gt_tl_end,
                "pred_tl_start": pred_tl_start,
                "pred_tl_end": pred_tl_end,
                "gt_precursor": gt_precursor,
                "pred_precursor": pred_precursor,
                "raw_response": raw_response,
            }
            enhanced_infos.append(info)

        metrics: Dict[str, float] = {}

        # --- Step1: Yes/No accuracy ---
        if s1_gts:
            s1_gts_arr = np.array(s1_gts, dtype=int)
            s1_preds_arr = np.array(s1_preds, dtype=int)
            acc_step1 = float(np.mean((s1_gts_arr == s1_preds_arr).astype(np.float32)))
            metrics["Step1_Acc"] = acc_step1
            metrics["Step1_Num"] = float(len(s1_gts))
        else:
            metrics["Step1_Acc"] = -1.0
            metrics["Step1_Num"] = 0.0

        # --- Step3: TL MAE + 1s Acc ---
        if tl_gt_starts:
            gt_s = np.array(tl_gt_starts, dtype=float)
            gt_e = np.array(tl_gt_ends, dtype=float)
            pred_s = np.array(tl_pred_starts, dtype=float)
            pred_e = np.array(tl_pred_ends, dtype=float)

            mae_start = float(np.mean(np.abs(pred_s - gt_s)))
            mae_end = float(np.mean(np.abs(pred_e - gt_e)))
            mae_mean = float((mae_start + mae_end) / 2.0)

            tol = 1.0
            acc_mask = (
                (np.abs(pred_s - gt_s) <= tol)
                & (np.abs(pred_e - gt_e) <= tol)
            )
            acc_1s = float(np.mean(acc_mask.astype(np.float32)))

            metrics["Step3_MAE_start"] = mae_start
            metrics["Step3_MAE_end"] = mae_end
            metrics["Step3_MAE_mean"] = mae_mean
            metrics["Step3_Acc_1s"] = acc_1s
            metrics["Step3_Num"] = float(len(gt_s))
        else:
            metrics["Step3_MAE_start"] = -1.0
            metrics["Step3_MAE_end"] = -1.0
            metrics["Step3_MAE_mean"] = -1.0
            metrics["Step3_Acc_1s"] = 0.0
            metrics["Step3_Num"] = 0.0

        # --- Step4: precursor MAE + 1s Acc ---
        if pre_gts:
            pre_gts_arr = np.array(pre_gts, dtype=float)
            pre_preds_arr = np.array(pre_preds, dtype=float)

            mae_pre = float(np.mean(np.abs(pre_preds_arr - pre_gts_arr)))

            tol = 1.0
            acc_pre = float(
                np.mean((np.abs(pre_preds_arr - pre_gts_arr) <= tol).astype(np.float32))
            )

            metrics["Step4_MAE"] = mae_pre
            metrics["Step4_Acc_1s"] = acc_pre
            metrics["Step4_Num"] = float(len(pre_gts_arr))
        else:
            metrics["Step4_MAE"] = -1.0
            metrics["Step4_Acc_1s"] = 0.0
            metrics["Step4_Num"] = 0.0

        # 总样本数
        metrics["Num_samples"] = float(len(self.data_dict))

        return metrics, enhanced_infos

    # ========= 工具函数 =========

    @staticmethod
    def _normalize_yes_no(text: str) -> Optional[str]:
        if not isinstance(text, str):
            return None
        t = text.strip().lower()
        if t.startswith("yes"):
            return "yes"
        if t.startswith("no"):
            return "no"
        # 有时候模型可能输出 "Yes." "No." "YES" 等
        m = re.search(r"\b(yes|no)\b", t)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _parse_time_range(text: str) -> Tuple[float, float]:
        """
        从字符串中解析 "start - end" 数字，返回 (start, end).
        """
        if not isinstance(text, str):
            return 0.0, 0.0

        m = re.search(r"([\d\.]+)\s*-\s*([\d\.]+)", text)
        if m:
            try:
                s = float(m.group(1))
                e = float(m.group(2))
                return s, e
            except Exception:
                pass

        nums = re.findall(r"[\d\.]+", text)
        if len(nums) >= 2:
            try:
                s = float(nums[0])
                e = float(nums[1])
                return s, e
            except Exception:
                pass

        return 0.0, 0.0

    @classmethod
    def _parse_time_range_allow_na(cls, text: str) -> Tuple[Optional[float], Optional[float]]:
        if not isinstance(text, str):
            return None, None
        if "n/a" in text.lower():
            return None, None
        s, e = cls._parse_time_range(text)
        return s, e

    @staticmethod
    def _parse_timestamp(text: str) -> float:
        if not isinstance(text, str):
            return 0.0
        nums = re.findall(r"[\d\.]+", text)
        if not nums:
            return 0.0
        try:
            return float(nums[0])
        except Exception:
            return 0.0

    @classmethod
    def _parse_timestamp_allow_na(cls, text: str) -> Optional[float]:
        if not isinstance(text, str):
            return None
        if "n/a" in text.lower():
            return None
        return cls._parse_timestamp(text)

    # ----- 从整段 CoT 输出中解析各 Step -----

    @classmethod
    def _parse_step1_from_response(cls, text: str) -> Optional[str]:
        """
        尽量从 response 中找到 Step1 的 Yes/No.
        优先匹配 "Step 1" 段，找不到则在全局里找第一个 yes/no.
        """
        if not isinstance(text, str):
            return None

        # 尝试截取 "Step 1" 区块
        m = re.search(r"(step\s*1[:\-]?.*?)(step\s*2[:\-]|step\s*3[:\-]|step\s*4[:\-]|$)",
                      text, flags=re.IGNORECASE | re.DOTALL)
        chunk = m.group(1) if m else text
        return cls._normalize_yes_no(chunk)

    @staticmethod
    def _parse_step2_from_response(text: str) -> Optional[str]:
        """
        Step2 是 caption，这里不做严格 NLP metric，只截出一个合理的片段，便于人工对比。
        简单策略：如果有 'Step 2'，取到下一步之前；否则返回 None.
        """
        if not isinstance(text, str):
            return None

        m = re.search(
            r"step\s*2[:\-]?(.*?)(step\s*3[:\-]|step\s*4[:\-]|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return None
        chunk = m.group(1).strip()
        # 防止太长，简单截断一下
        if len(chunk) > 512:
            chunk = chunk[:512]
        return chunk if chunk else None

    @classmethod
    def _parse_step3_from_response(cls, text: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Step3: crash temporal range.
        优先在 'Step 3' 段中找 "start - end"；找不到就全局 fallback.
        如果模型输出 "N/A" 之类，则返回 (None, None).
        """
        if not isinstance(text, str):
            return None, None

        if "n/a" in text.lower():
            return None, None

        # 尝试局部解析 Step 3 区块
        m = re.search(
            r"step\s*3[:\-]?(.*?)(step\s*4[:\-]|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        chunk = m.group(1) if m else text

        if "n/a" in chunk.lower():
            return None, None

        s, e = cls._parse_time_range(chunk)
        return s, e

    @classmethod
    def _parse_step4_from_response(cls, text: str) -> Optional[float]:
        """
        Step4: precursor timestamp.
        优先在 'Step 4' 段内找第一个数字；找不到就全局 fallback.
        如果模型输出 "N/A" 则返回 None.
        """
        if not isinstance(text, str):
            return None

        if "n/a" in text.lower():
            return None

        m = re.search(
            r"step\s*4[:\-]?(.*)$",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        chunk = m.group(1) if m else text

        if "n/a" in chunk.lower():
            return None

        return cls._parse_timestamp(chunk)
