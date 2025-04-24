import os
import torch
import librosa
import numpy as np
from datasets import Dataset, Audio
from typing import Dict, List, Union, Any, Tuple, Optional, Callable
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
from tqdm import tqdm

def load_data_from_directory(data_dir: str) -> Dataset:
    """
    지정된 디렉토리에서 오디오-텍스트 쌍을 로드합니다.
    파일 구조가 다음과 같이 되어 있다고 가정합니다:
    - data_dir/wav와 data_dir/label
    또는
    - data_dir/train/wav와 data_dir/train/label
    - data_dir/val/wav와 data_dir/val/label
    
    오디오와 텍스트 파일은 공통 하위 경로와 파일명으로 매칭됩니다.
    오디오 파일은 TS_로 시작하고 텍스트 파일은 TL_로 시작합니다.
    """
    audio_files: List[str] = []
    transcripts: List[str] = []
    
    # 가능한 디렉토리 구조 확인
    data_subdirs = []
    
    # 기본 구조: data_dir/wav와 data_dir/label
    if os.path.exists(os.path.join(data_dir, "wav")) and os.path.exists(os.path.join(data_dir, "label")):
        data_subdirs.append(("", os.path.join(data_dir, "wav"), os.path.join(data_dir, "label")))
    
    # train/val 구조 확인
    for subdir in ["train", "val"]:
        wav_path = os.path.join(data_dir, subdir, "wav")
        label_path = os.path.join(data_dir, subdir, "label")
        
        # 폴더 존재 확인
        if os.path.exists(wav_path) and os.path.exists(label_path):
            
            wav_files_exist = False
            for root, _, files in os.walk(wav_path):
                if any(file.endswith('.wav') for file in files):
                    wav_files_exist = True
                    break
            
            # label 폴더 내 .txt 파일 확인
            txt_files_exist = False
            for root, _, files in os.walk(label_path):
                if any(file.lower().endswith('.txt') for file in files):
                    txt_files_exist = True
                    break
            
            # 파일이 없으면 val 폴더 스킵
            if not (wav_files_exist and txt_files_exist):
                print(f"'{subdir}' 폴더에 .wav 또는 .txt 파일이 없습니다. 스킵합니다.")
                continue
            
            # 모든 조건을 만족하면 처리 대상에 추가
            data_subdirs.append((subdir, wav_path, label_path))
    
    if not data_subdirs:
        print("경고: 유효한 데이터 디렉토리 구조를 찾을 수 없습니다. 기본 경로를 사용합니다.")
        data_subdirs.append(("", data_dir, data_dir))
    
    # 각 유효한 디렉토리 쌍에서 데이터 로드 (tqdm 진행바 추가)
    for subdir_name, wav_dir, label_dir in tqdm(data_subdirs, desc="디렉토리 처리 중", unit="디렉토리"):
        print(f"'{subdir_name}' 데이터 디렉토리에서 파일 로드 중...")
        
        # 오디오 파일 매핑 생성
        audio_paths = {}
        total_wav_files = 0
        
        # 오디오 파일 찾기 - 파일 수를 먼저 계산하여 tqdm에 전달
        wav_files_count = 0
        for _, _, files in os.walk(wav_dir):
            wav_files_count += sum(1 for file in files if file.endswith('.wav'))
        
        # 오디오 파일 처리에 진행바 추가
        wav_file_progress = tqdm(total=wav_files_count, desc="오디오 파일 처리 중", unit="파일")
        for root, _, files in os.walk(wav_dir):
            for file in files:
                if file.endswith('.wav'):
                    # 전체 경로에서 wav_dir 이후 부분을 키로 사용
                    rel_path = os.path.relpath(root, wav_dir)
                    if rel_path == '.':
                        key = file
                    else:
                        key = os.path.join(rel_path, file)
                    
                    # 오디오 파일 경로를 패턴 식별 키로 변환
                    # TS_D01\D01\J020\S000781\0003.wav -> D01\J020\S000781\0003
                    path_parts = key.split(os.sep)
                    if len(path_parts) >= 2 and path_parts[0].startswith('TS_'):
                        # TS_ 접두사를 제거하고 첫 번째 부분을 제거 (TS_D01)
                        pattern_key = os.path.join(*path_parts[1:])
                        # 확장자 제거
                        pattern_key = os.path.splitext(pattern_key)[0]
                        audio_paths[pattern_key] = os.path.join(root, file)
                    else:
                        # 일반적인 패턴이 아닌 경우 그대로 사용
                        pattern_key = os.path.splitext(key)[0]  # 확장자 제거
                        audio_paths[pattern_key] = os.path.join(root, file)
                    
                    total_wav_files += 1
                    wav_file_progress.update(1)
        wav_file_progress.close()
        
        # 텍스트 파일 경로 매핑 구축 (패턴 키 -> 텍스트 파일 경로)
        text_paths = {}
        
        # 텍스트 파일 찾기
        for root, _, files in os.walk(label_dir):
            for file in files:
                # 텍스트 파일은 .txt 확장자만 처리
                if not file.lower().endswith('.txt'):
                    continue
                
                # 전체 경로에서 label_dir 이후 부분을 키로 사용
                rel_path = os.path.relpath(root, label_dir)
                if rel_path == '.':
                    label_key = file
                else:
                    label_key = os.path.join(rel_path, file)
                
                # 텍스트 파일 경로를 패턴 식별 키로 변환
                # TL_D01\D01\J020\S000781\0003.txt -> D01\J020\S000781\0003
                path_parts = label_key.split(os.sep)
                if len(path_parts) >= 2 and path_parts[0].startswith('TL_'):
                    # TL_ 접두사를 제거하고 첫 번째 부분을 제거 (TL_D01)
                    pattern_key = os.path.join(*path_parts[1:])
                    # 확장자 제거
                    pattern_key = os.path.splitext(pattern_key)[0]
                else:
                    # 일반적인 패턴이 아닌 경우 그대로 사용
                    pattern_key = os.path.splitext(label_key)[0]  # 확장자 제거
                
                text_paths[pattern_key] = os.path.join(root, file)
        
        # 이제 각 오디오 파일에 대해 매칭되는 텍스트 파일 찾기
        matched_count = 0
        failed_matches = []
        
        # WAV 파일 매칭 진행 상태 표시
        wav_matching_progress = tqdm(total=len(audio_paths), desc="WAV 파일 매칭 중", unit="파일")
        
        for pattern_key, audio_path in audio_paths.items():
            wav_matching_progress.update(1)
            
            if pattern_key in text_paths:
                try:
                    # 텍스트 파일 읽기
                    with open(text_paths[pattern_key], 'r', encoding='utf-8') as f:
                        transcript = f.read().strip()
                    
                    # 디버깅 정보 출력
                    if matched_count < 3:  # 처음 3개만 출력
                        print(f"매칭 성공: 패턴 키 '{pattern_key}'")
                        print(f"  오디오: {audio_path}")
                        print(f"  라벨: {text_paths[pattern_key]}")
                        print(f"  텍스트: {transcript[:50]}..." if len(transcript) > 50 else transcript)
                    
                    audio_files.append(audio_path)
                    transcripts.append(transcript)
                    matched_count += 1
                    
                    # 매칭 정보 업데이트 - WAV 파일 기준으로 매칭된 비율 표시
                    match_ratio = (matched_count / wav_matching_progress.n) * 100
                    wav_matching_progress.set_postfix(match_ratio=f"{match_ratio:.2f}%", matched=matched_count, processed=wav_matching_progress.n)
                except Exception as e:
                    print(f"파일 {text_paths[pattern_key]}을 읽는 중 오류 발생: {e}")
            else:
                # 매칭 실패한 항목 기록
                print(f"매칭 실패: 패턴 키 '{pattern_key}', 오디오 파일: {audio_path}")
                failed_matches.append((pattern_key, audio_path))
        
        wav_matching_progress.close()
        
        print(f"'{subdir_name}' 디렉토리에서 {len(audio_paths)} 오디오 파일 중 {matched_count} 매칭된 쌍을 찾았습니다.")
        print(f"매칭 비율: {(matched_count / len(audio_paths) * 100):.2f}% (WAV 파일 기준)")
        
        # 매칭 실패 항목 요약 출력
        if failed_matches:
            failed_count = len(failed_matches)
            print(f"매칭 실패: {failed_count}개의 오디오 파일에 대응하는 텍스트 파일을 찾지 못했습니다.")
            
            # 첫 10개 실패 항목만 상세 출력 (너무 많으면 출력이 너무 길어질 수 있음)
            print("실패 항목 샘플 (최대 10개):")
            for i, (pattern_key, audio_path) in enumerate(failed_matches[:10]):
                print(f"  {i+1}. 패턴 키: '{pattern_key}' (오디오 파일: {audio_path})")
            
            # 10개 이상인 경우 생략 표시
            if failed_count > 10:
                print(f"  ... 그 외 {failed_count - 10}개 항목 생략...")
    
    print(f"총 {len(audio_files)}개의 오디오-텍스트 쌍을 찾았습니다.")
    if len(audio_files) == 0:
        print("경고: 매칭되는 오디오-텍스트 쌍이 없습니다. 데이터 경로를 확인하세요.")
    else:
        print(f"첫 번째 오디오 파일: {audio_files[0]}")
        print(f"첫 번째 텍스트: {transcripts[0][:50]}..." if len(transcripts[0]) > 50 else transcripts[0])
    
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