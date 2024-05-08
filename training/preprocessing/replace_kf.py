from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import json
import torch

system_message = """你是蜀门游戏助手，你需要回答用户提出的和蜀门相关的问题"""
model_id = "shenzhi-wang/Llama3-8B-Chinese-Chat"

torch_dtype = torch.bfloat16
quant_storage_dtype = torch.bfloat16
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_quant_storage=quant_storage_dtype,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto",quantization_config=quantization_config
)


def get_response(output):
    if "assistant\n\n" in output:
        parts = output.split("assistant\n\n", 1)
        return parts[1]
    else:
        return output
def get_gpu_memory_usage():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("CUDA device detected.")
        print("Device name:", torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0)/1024**3,1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0)/1024**3,1), "GB")
def contain_image(messages):
    # return False
    for item in messages:
        if "https://mp.weixin.qq.com" in item["content"]:
            print("contain_image")
            return True
    return False
def generate_question(sample):
    print("resolve:")
    print(sample["resolve"])
    if not sample["resolve"] or not contain_image(sample["messages"]):
        sample["replaced"] = json.dumps(sample["messages"], ensure_ascii=False)
    else:
        messages = [
            {"role": "system", "content":"下面是一段客服对话记录，其中有部分以https://mp.weixin.qq.com开头的是用户发送的图片，请根据上下文猜测图片的内容，生成一段文字内容替换图片，看起来就像用户发的是文字，让整个对话看起来真实合理，输出修改过的json。"},
            {"role": "user", "content": sample["messages"]},
        ]
        # chat = tokenizer.apply_chat_template(
        #     messages, add_generation_prompt=True, return_tensors="pt", tokenize=False
        # )
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        
        # print(chats) 
        # input_ids = tokenizer(chat, return_tensors="pt", padding=True).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=2048,
        )
        response = outputs[0]
        response = tokenizer.decode(response, skip_special_tokens=True)
        response = get_response(response)
        print(response)
        get_gpu_memory_usage()
        torch.cuda.empty_cache()
        sample["replaced"] = response

    return sample

# 处理客服聊天记录
chat_dataset = load_dataset(
        "ytcheng/sm_kf"
)
# chat_dataset = chat_dataset.sort("time")
print(chat_dataset)
print(chat_dataset["train"][0])

chat_dataset = chat_dataset.map(generate_question)
print(chat_dataset)
chat_dataset.push_to_hub("ytcheng/sm_kf")