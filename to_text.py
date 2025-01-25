import sounddevice as sd
import numpy as np
import whisper
import wave
import tempfile
import os

def record_audio(duration=5, sample_rate=16000):
    """录制固定时长的音频"""
    print("开始录音...")
    audio = sd.rec(int(duration * sample_rate), 
                  samplerate=sample_rate, 
                  channels=1,
                  dtype=np.int16)
    sd.wait()
    print("录音结束")
    return audio

def save_audio(audio, filename, sample_rate=16000):
    """保存音频到WAV文件"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

def transcribe_audio(audio_path):
    """使用Whisper转录音频"""
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def record_and_transcribe(duration=5):
    """录音并转录"""
    # 录制音频
    audio = record_audio(duration)
    
    # 创建临时文件保存音频
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        save_audio(audio, temp_path)
    
    try:
        # 转录音频
        text = transcribe_audio(temp_path)
        return text
    finally:
        # 清理临时文件
        os.unlink(temp_path)