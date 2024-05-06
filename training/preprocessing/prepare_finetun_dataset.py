from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from collections import defaultdict
import time

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

# 处理问答对
def create_conversation(sample):
    sample["messages"] = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role":"user", 
            "content": sample["question"]
        }, 
        {
            "role":"assistant", 
            "content":sample["answer"]
        }
    ]
    return sample
chat_dataset = load_dataset(
        "ytcheng/sm_question"
)
columns_to_remove = list(chat_dataset["train"].features)

chat_dataset = chat_dataset.map(create_conversation, remove_columns=columns_to_remove,batched=False)
dataset = concatenate_datasets([kf_dataset, chat_dataset["train"]])
dataset = DatasetDict({"train":  dataset})
dataset.push_to_hub("ytcheng/sm_kf_chat_message")
# print(result)
# print(result[0])
# print(result[1])
# print(result[2])
# print(result[3])