from .activitynet_qa import ActivitynetQADataset
from .base import BaseEvalDataset
from .charades_sta import CharadesSTADataset
from .egoschema import EgoSchemaDataset
from .longvideobench import LongVideoBenchDataset
from .lvbench import LVBenchDataset
from .mlvu import MLVUDataset
from .mmvu import MMVUDataset
from .mvbench import MVBenchDataset
from .nextqa import NextQADataset
from .perception_test import PerceptionTestDataset
from .tempcompass import TempCompassDataset
from .videomme import VideoMMEDataset

from .ai2d import AI2DDataset
from .chartqa import ChartQADataset
from .docvqa import DocVQADataset
from .mathvista import MathVistaDataset
from .mmmu import MMMUDataset
from .ocrbench import OCRBenchDataset
from .gqa import GQADataset
from .mmmupro import MMMUProDataset
from .realworldqa import RealWorldQADataset
from .blink import BLINKDataset
from .mme import MMEDataset
from .infovqa import InfoVQADataset
from .mathverse import MathVerseDataset
from .mathvision import MathVisionDataset

from .crash_video_tl import CrashTemporalLocalizationDataset
from .crash_video_ar import CrashAccidentRecognitionDataset
from .crash_video_ac import CrashAccidentCaptionDataset
from .crash_video_pl import CrashPrecursorLocalizationDataset
from .crash_video_am import CrashAccidentMeasureDataset
from .crash_video_acau import CrashAccidentCauseDataset
from .crash_video_al import CrashAnticipationLocalizationDataset
from .crash_video_cot import CrashCoTDataset

DATASET_REGISTRY = {
    "videomme": VideoMMEDataset,
    "mmvu": MMVUDataset,
    "mvbench": MVBenchDataset,
    "egoschema": EgoSchemaDataset,
    "perception_test": PerceptionTestDataset,
    "activitynet_qa": ActivitynetQADataset,
    "mlvu": MLVUDataset,
    "longvideobench": LongVideoBenchDataset,
    "lvbench": LVBenchDataset,
    "tempcompass": TempCompassDataset,
    "nextqa": NextQADataset,
    "charades_sta": CharadesSTADataset,
    "AI2D": AI2DDataset,
    "ChartQA": ChartQADataset,
    "DocVQA": DocVQADataset,
    "MathVista": MathVistaDataset,
    "MMMU": MMMUDataset,
    "OCRBench": OCRBenchDataset,
    "GQA": GQADataset,
    "MMMU_Pro": MMMUProDataset,
    "RealWorldQA": RealWorldQADataset,
    "BLINK": BLINKDataset,
    "MME": MMEDataset,
    "InfoVQA": InfoVQADataset,
    "MathVerse": MathVerseDataset,
    "MathVision": MathVisionDataset,
    "crash_tl": CrashTemporalLocalizationDataset,
    "crash_ar": CrashAccidentRecognitionDataset,
    "crash_ac": CrashAccidentCaptionDataset,
    "crash_pl": CrashPrecursorLocalizationDataset,
    "crash_cot": CrashCoTDataset,
    "crash_am": CrashAccidentMeasureDataset,
    "crash_acau": CrashAccidentCauseDataset,
    "crash_al": CrashAnticipationLocalizationDataset,
}


def build_dataset(benchmark_name: str, **kwargs) -> BaseEvalDataset:
    assert benchmark_name in DATASET_REGISTRY, (
        f"Unknown benchmark: {benchmark_name}, available: {DATASET_REGISTRY.keys()}"
    )
    return DATASET_REGISTRY[benchmark_name](**kwargs)
