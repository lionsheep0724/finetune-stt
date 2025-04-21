import evaluate
import numpy as np
from typing import Dict, List, Any, Union, Optional, Callable
from transformers import WhisperTokenizer
from transformers.trainer_utils import EvalPrediction

# 메트릭 로드
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
    """
    예측 결과를 바탕으로 WER과 CER을 계산합니다.
    """
    pred_ids: np.ndarray = pred.predictions
    label_ids: np.ndarray = pred.label_ids

    # -100을 패드 토큰 ID로 대체
    label_ids[label_ids == -100] = pred.tokenizer.pad_token_id

    # 토큰을 그룹화하지 않고 메트릭을 계산
    pred_str: List[str] = pred.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str: List[str] = pred.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # WER 계산 (단어 단위 오류율)
    wer: float = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    
    # CER 계산 (문자 단위 오류율)
    cer: float = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    return {
        "wer": wer,
        "cer": cer
    }

class MetricsCallback:
    """
    평가 중에 메트릭을 기록하기 위한 콜백
    """
    tokenizer: WhisperTokenizer
    
    def __init__(self, tokenizer: WhisperTokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, pred: EvalPrediction) -> Dict[str, float]:
        pred.tokenizer = self.tokenizer
        return compute_metrics(pred) 