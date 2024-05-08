from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import time
import torch

system_message = """你是蜀门游戏助手，你需要回答用户提出的和蜀门相关的问题"""
# 处理客服聊天记录
chat_dataset = load_dataset(
        "ytcheng/sm_kf_chat"
)
chat_dataset = chat_dataset.sort("time")
print(chat_dataset)
print(chat_dataset["train"][0])

grouped_data = defaultdict(list)


for entry in chat_dataset["train"]:
    # print(entry)
    openid = entry['openid']
    grouped_data[openid].append(entry)
# print(grouped_data)
result = []

# 处理每个 openid 的消息
for openid, messages in grouped_data.items():
    # 按照时间戳排序消息
    sorted_messages = sorted(messages, key=lambda x: x['time'])
    
    conversation = [{
        "role": "system",
        "content": system_message
    }]
    for message in sorted_messages:
        timeArray = time.localtime(message["time"])
        otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        if message['opercode'] == 2003:  # 用户消息
            conversation.append({"role": "user", "content": message['text']})
        elif message['opercode'] == 2002 and "https://www.wjx.top" not in message['text']:  # 客服消息
            conversation.append({ "role": "assistant", "content": message['text']})
    
    result.append({"messages": conversation})
kf_dataset = Dataset.from_list(result)


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

def generate_question(batch):
    # print(batch)
    chats = []
    for index, messages in enumerate(batch["messages"]):

        messages = [
            {"role": "system", "content":"下面是一段客服对话记录，你判断是否已经解决了用户的问题或者给用户给供了有价值的信息，如果解决了问题或提供了有价值的信息返回true,未解决返回false"},
            {"role": "user", "content": messages},
        ]
        chat = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", tokenize=False
        )
        chats.append(chat)
    # print(chats) 
    input_ids = tokenizer(chats, return_tensors="pt", padding=True).to(model.device)

    outputs = model.generate(
        **input_ids,
        max_new_tokens=512,
        # temperature=0.5,
        # top_p=0.9,
    )
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results = map(get_response, results)
    results = list(results)

    resolve = []
    for item in results:
        if "false" in item or "False" in item:
            resolve.append(False)
        else:
            resolve.append(True)
    batch["resolve"] = resolve
    batch["output"] = results
    print(batch)
    input_ids.to("cpu")
    torch.cuda.empty_cache()
    get_gpu_memory_usage()

    return batch

kf_dataset.map(generate_question, batched=True, batch_size=4)
kf_dataset.push_to_hub("ytcheng/sm_kf")