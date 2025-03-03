from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM with specific instructions on translation.'''
    return f"""# 任務說明
你是一台翻譯機，如果你拿到文言文，你要將文言文翻譯成現代文，如果你拿到現代文，你要將現代文翻譯成文言文，其中文言文又稱為古文，現代文又稱為白話文。請注意：
1. 僅需根據指令給出翻譯，不要生成任何其他內容。
2. 僅提供單一版本的翻譯結果。
# 使用者問題
{instruction}
#回答
"""

def get_bnb_config() -> BitsAndBytesConfig:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    return bnb_config

