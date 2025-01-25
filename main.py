from utils import load_apikey
from to_text import record_and_transcribe
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import asyncio
import edge_tts
import tempfile
import os
import sounddevice as sd
import soundfile as sf

async def text_to_speech(text, voice="zh-CN-XiaoxiaoNeural"):
    """将文本转换为语音"""
    communicate = edge_tts.Communicate(text, voice)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        
    try:
        await communicate.save(temp_path)
        
        # 使用sounddevice播放音频
        data, samplerate = sf.read(temp_path)
        sd.play(data, samplerate)
        sd.wait()  # 等待播放完成
            
    finally:
        os.unlink(temp_path)

def get_local_llm_response(text):
    """使用本地LLM获取回复"""
    try:
        # 加载FP16模型和分词器
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen1.5-1.8B",  # 切换到Qwen1.5-1.8B
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B", trust_remote_code=True)
        
        # 构建提示模板
        prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        
        # 生成回复
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)  # 设为False以保留特殊标记
        
        # 简化提取助手回复的逻辑
        parts = response.split("<|im_start|>assistant\n")
        if len(parts) > 1:
            response = parts[1].split("<|im_end|>")[0].strip()
        else:
            response = response.strip()
            
        return response
    except Exception as e:
        print(f"获取AI回复时出错: {e}")
        return None

async def main():
    # 录制音频并转录
    text = record_and_transcribe(duration=5)
    print("语音转文字:", text)
    
    # 获取AI回复
    if text:
        response = get_local_llm_response(text)
        print("AI回复:", response)
        # 将回复转换为语音
        await text_to_speech(response)

if __name__ == "__main__":
    asyncio.run(main())