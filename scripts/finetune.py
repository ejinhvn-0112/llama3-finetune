import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 외장 SSD의 캐시 및 저장 디렉토리 설정
cache_dir = "/Volumes/T7 Shield/cache"
save_dir = "/Volumes/T7 Shield/saved_model"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

# 환경 변수 설정
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir

# 환경 변수 확인
print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")
print(f"HF_HOME: {os.environ.get('HF_HOME')}")

# 모델과 토크나이저 로드
print("토크나이저를 로드합니다...")
tokenizer = AutoTokenizer.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B", cache_dir=cache_dir)
print("토크나이저 로드 완료.")

print("모델을 로드합니다...")
model = AutoModelForCausalLM.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B", cache_dir=cache_dir)
print("모델 로드 완료")

# 모델과 토크나이저 저장
print("모델과 토크나이저를 저장합니다...")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"모델과 토크나이저가 {save_dir}에 저장되었습니다.")

# 세마포어 정리
if 'posix' in os.name:
    import resource

    print("Cleaning up semaphores...")
    semaphores = resource.getrlimit(resource.RLIMIT_NPROC)
    print(f"Semaphores before cleanup: {semaphores}")

    # 현재 프로세스의 리소스 제한 확인
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NPROC)
    resource.setrlimit(resource.RLIMIT_NPROC, (soft_limit, hard_limit))

    print("Semaphores cleaned up.")
