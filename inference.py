#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import librosa
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whisper v3 Turbo 파인튜닝 모델을 이용한 음성 인식")
    parser.add_argument("--model_path", type=str, required=True, help="파인튜닝된 모델의 경로")
    parser.add_argument("--audio_path", type=str, required=True, help="변환할 오디오 파일 경로")
    parser.add_argument("--output_file", type=str, default=None, help="결과를 저장할 텍스트 파일 경로 (지정하지 않으면 콘솔에 출력)")
    parser.add_argument("--chunk_length", type=int, default=None, help="오디오 청크 길이(초), 지정하지 않으면 모델 기본값 사용")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    # 모델 및 프로세서 로드
    print(f"모델 로드 중: {args.model_path}")
    
    # 일반적인 방법으로 로드
    if args.chunk_length is None:
        processor: WhisperProcessor = WhisperProcessor.from_pretrained(args.model_path)
    else:
        # chunk_length를 사용자 지정 값으로 설정하여 로드
        print(f"chunk_length {args.chunk_length}초로 설정")
        feature_extractor: WhisperFeatureExtractor = WhisperFeatureExtractor.from_pretrained(
            args.model_path,
            chunk_length=args.chunk_length
        )
        tokenizer: WhisperTokenizer = WhisperTokenizer.from_pretrained(args.model_path)
        processor: WhisperProcessor = WhisperProcessor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )
    
    model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(args.model_path)
    
    # GPU 사용 가능 시 GPU로 이동
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 오디오 로드 및 전처리
    print(f"오디오 로드 중: {args.audio_path}")
    # 오디오 파일 읽기 및 리샘플링 (16kHz)
    audio: np.ndarray
    sr: int
    audio, sr = librosa.load(args.audio_path, sr=16000)
    
    # 오디오 실제 길이 계산
    actual_length_seconds: float = len(audio) / sr
    print(f"오디오 길이: {actual_length_seconds:.2f}초")
    
    # 오디오 특징 추출
    input_features: torch.Tensor = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    
    # 추론
    print("음성 인식 중...")
    with torch.no_grad():
        generated_ids: torch.Tensor = model.generate(input_features)
    
    # 디코딩
    transcription: str = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 결과 출력 또는 저장
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(transcription)
        print(f"결과가 {args.output_file}에 저장되었습니다.")
    else:
        print("\n인식 결과:")
        print("-" * 50)
        print(transcription)
        print("-" * 50)

if __name__ == "__main__":
    main() 