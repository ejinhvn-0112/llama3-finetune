import os
from fastapi import FastAPI, Request
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

# FastAPI 앱 생성
app = FastAPI()

# 모델과 토크나이저 로드
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(save_dir)
print("Tokenizer loaded.")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(save_dir)
print("Model loaded.")


@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data['prompt']

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # pad_token_id를 eos_token_id로 설정
    pad_token_id = tokenizer.eos_token_id

    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=50, pad_token_id=pad_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"generated_text": generated_text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
