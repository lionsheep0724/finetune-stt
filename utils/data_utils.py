import os
import torch
import librosa
import numpy as np
from datasets import Dataset, Audio
from typing import Dict, List, Union, Any, Tuple, Optional, Callable
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer

def load_data_from_directory(data_dir: str) -> Dataset:
    """
    .wav와 .txt 파일 쌍으로 구성된 데이터셋을 로드합니다.
    파일 이름은 같고 확장자만 다르다고 가정합니다.
    """
    audio_files: List[str] = []
    transcripts: List[str] = []
    
    # 디렉토리 내의 모든 파일 탐색
    for file in os.listdir(data_dir):
        if file.endswith('.wav'):
            base_name: str = os.path.splitext(file)[0]
            txt_file: str = f"{base_name}.txt"
            
            if os.path.exists(os.path.join(data_dir, txt_file)):
                audio_path: str = os.path.join(data_dir, file)
                transcript_path: str = os.path.join(data_dir, txt_file)
                
                # 텍스트 파일 읽기
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript: str = f.read().strip()
                
                audio_files.append(audio_path)
                transcripts.append(transcript)
    
    # 데이터셋 생성
    dataset: Dataset = Dataset.from_dict({
        "audio": audio_files,
        "sentence": transcripts
    })
    
    # Audio 객체로 변환
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    return dataset

def prepare_dataset(batch: Dict[str, Any], processor: WhisperProcessor) -> Dict[str, Any]:
    """
    오디오 데이터를 특징 벡터로 변환하고 텍스트를 토큰화합니다.
    """
    # 오디오 로드 및 16kHz로 리샘플링
    audio: Dict[str, Union[np.ndarray, int]] = batch["audio"]
    
    # 입력 특징 추출
    batch["input_features"] = processor.feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # 타겟 텍스트를 라벨 ID로 인코딩
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    
    return batch

def split_dataset(dataset: Dataset, test_size: float = 0.1, seed: int = 42) -> Dict[str, Dataset]:
    """
    데이터셋을 학습용과 테스트용으로 분할합니다.
    """
    # 데이터셋 섞기 및 분할
    dataset = dataset.shuffle(seed=seed)
    
    # 테스트 세트의 샘플 수 계산
    test_samples: int = int(len(dataset) * test_size)
    train_samples: int = len(dataset) - test_samples
    
    # 분할
    train_dataset: Dataset = dataset.select(range(train_samples))
    test_dataset: Dataset = dataset.select(range(train_samples, len(dataset)))
    
    return {
        "train": train_dataset,
        "test": test_dataset
    }

class DataCollatorSpeechSeq2SeqWithPadding:
    """
    배치 내의 오디오 특징과 라벨을 패딩하는 데이터 콜레이터
    """
    processor: Any
    decoder_start_token_id: int
    
    def __init__(self, processor: Any, decoder_start_token_id: int):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id
        
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 입력과 라벨을 분리 (서로 다른 길이를 가지므로 다른 패딩 방법이 필요)
        input_features: List[Dict[str, Union[List[int], torch.Tensor]]] = [{"input_features": feature["input_features"]} for feature in features]
        batch: Dict[str, torch.Tensor] = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # 토큰화된 라벨 시퀀스 가져오기
        label_features: List[Dict[str, List[int]]] = [{"input_ids": feature["labels"]} for feature in features]
        # 라벨을 최대 길이로 패딩
        labels_batch: Dict[str, torch.Tensor] = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # 패딩을 -100으로 대체하여 손실 계산 시 올바르게 무시
        labels: torch.Tensor = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # bos 토큰이 이전 토큰화 단계에서 추가된 경우,
        # 나중에 어차피 추가되므로 여기서 잘라냄
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
            
        batch["labels"] = labels
        
        return batch 