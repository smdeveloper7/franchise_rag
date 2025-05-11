#!/bin/bash

JSON_PATH=$1
DEVICE=${2:-cuda}

if [ -z "$JSON_PATH" ]; then
  echo "❗ 테스트 JSON 파일 경로를 입력하세요."
  echo "예시: bash franchise_RAG.sh ./data/test/test_01.json [cuda|cpu]"
  exit 1
fi

echo "지식베이스 생성 중입니다..."
python3 create_collection.py --json_path "$JSON_PATH" --device "$DEVICE"
if [ $? -ne 0 ]; then
  echo "❌ 지식베이스 생성 중 오류가 발생했습니다. 스크립트를 종료합니다."
  exit 1
fi

echo "지식베이스 생성 완료"

echo "💡 추론 시작 중..."
python3 run_inference.py --json_path "$JSON_PATH" --device "$DEVICE"
if [ $? -ne 0 ]; then
  echo "❌ 추론 실행 중 오류가 발생했습니다. 스크립트를 종료합니다."
  exit 1
fi
