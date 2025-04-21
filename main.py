#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Dict, List, Optional, Union, Any, Tuple
from datasets import DatasetDict, Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from utils.data_utils import (
    load_data_from_directory,
    prepare_dataset,
    split_dataset,
    DataCollatorSpeechSeq2SeqWithPadding
)
from utils.metrics import MetricsCallback

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whisper v3 Turbo 파인튜닝")
    parser.add_argument("--data_dir", type=str, default="./data", help="데이터셋 디렉토리 경로")
    parser.add_argument("--output_dir", type=str, default="./whisper-v3-turbo-ko", help="모델 저장 디렉토리")
    parser.add_argument("--batch_size", type=int, default=8, help="배치 사이즈")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="평가 배치 사이즈")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="학습률")
    parser.add_argument("--max_steps", type=int, default=5000, help="최대 학습 스텝")
    parser.add_argument("--warmup_steps", type=int, default=500, help="웜업 스텝 수")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="그래디언트 누적 스텝 수")
    parser.add_argument("--eval_steps", type=int, default=1000, help="평가 간격")
    parser.add_argument("--save_steps", type=int, default=1000, help="저장 간격")
    parser.add_argument("--fp16", action="store_true", help="fp16 사용")
    parser.add_argument("--test_size", type=float, default=0.1, help="테스트 데이터 비율")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--num_proc", type=int, default=4, help="데이터 전처리 프로세스 수")
    parser.add_argument("--chunk_length", type=int, default=30, help="오디오 청크 길이(초), 기본값은 30초")
    parser.add_argument("--logging_steps", type=int, default=25, help="로깅 간격")
    parser.add_argument("--tensorboard_dir", type=str, default=None, help="TensorBoard 로그 디렉토리 (기본값: output_dir)")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    # TensorBoard 로그 디렉토리 설정
    tensorboard_dir: str = args.tensorboard_dir if args.tensorboard_dir else args.output_dir
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # 데이터 로드
    print("데이터 로드 중...")
    dataset: Dataset = load_data_from_directory(args.data_dir)
    
    # 학습/테스트 데이터 분할
    print(f"총 {len(dataset)} 개의 샘플이 로드되었습니다. 데이터 분할 중...")
    dataset_dict: Dict[str, Dataset] = split_dataset(dataset, test_size=args.test_size, seed=args.seed)
    
    # 모델 및 프로세서 로드
    print("Whisper v3 Turbo 모델 로드 중...")
    model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    
    # chunk_length를 사용하여 feature_extractor 생성
    print(f"chunk_length {args.chunk_length}초로 Feature Extractor 초기화...")
    feature_extractor: WhisperFeatureExtractor = WhisperFeatureExtractor.from_pretrained(
        "openai/whisper-large-v3",
        chunk_length=args.chunk_length
    )
    tokenizer: WhisperTokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-large-v3", 
        language="Korean", 
        task="transcribe"
    )
    processor: WhisperProcessor = WhisperProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
    # 언어 및 태스크 설정
    model.generation_config.language = "korean"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    
    # 데이터셋 전처리
    print("데이터셋 전처리 중...")
    def map_function(batch: Dict[str, Any]) -> Dict[str, Any]:
        return prepare_dataset(batch, processor)
    
    dataset_dict_hf: DatasetDict = DatasetDict({
        "train": dataset_dict["train"],
        "test": dataset_dict["test"]
    })
    
    processed_dataset: DatasetDict = dataset_dict_hf.map(
        map_function,
        remove_columns=dataset_dict_hf["train"].column_names,
        num_proc=args.num_proc
    )
    
    # 데이터 콜레이터 생성
    data_collator: DataCollatorSpeechSeq2SeqWithPadding = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )
    
    # 메트릭 콜백 생성
    metrics_callback: MetricsCallback = MetricsCallback(processor.tokenizer)
    
    # 학습 인자 설정
    training_args: Seq2SeqTrainingArguments = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=args.fp16,
        evaluation_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_dir=tensorboard_dir,
    )
    
    print(f"TensorBoard 로그는 {tensorboard_dir} 디렉토리에 저장됩니다")
    print(f"학습 진행 상황을 보려면 'docker-compose up -d'를 실행하고 http://localhost:6006 으로 접속하세요")
    
    # 트레이너 설정
    trainer: Seq2SeqTrainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        data_collator=data_collator,
        compute_metrics=metrics_callback,
        tokenizer=processor.feature_extractor,
    )
    
    # 학습 시작
    print("학습 시작...")
    trainer.train()
    
    # 모델 저장
    print(f"학습 완료. 모델 저장 중: {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    print("완료!")

if __name__ == "__main__":
    main() 