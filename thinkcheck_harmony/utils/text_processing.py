'''文本预处理工具'''
import re

def split_sentences(text: str):
    text = text.replace('；', '。').replace('！', '。').replace('？', '。')
    return [s.strip() for s in text.split('。') if s.strip()]
