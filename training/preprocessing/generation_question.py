from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch
from tqdm import tqdm
import json
import re
from datasets import Dataset, DatasetDict


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
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto",quantization_config=quantization_config
)

raw_datasets = load_dataset("ytcheng/sm_news")
print(raw_datasets)
drug_dataset = raw_datasets.filter(lambda x: x["content"]!="").filter(lambda x: x["type"]==5)
print(drug_dataset)

generate_data = []
chats = []
batch_size = 32
total_data = len(drug_dataset['train']) * 10

with tqdm(total=total_data) as pbar:
    for i in range(1, 11):
        for data in drug_dataset['train']:
            messages = [
                {"role": "system", "content":"从用户发的文本中，提取出尽可能多的问题和答案，以json格式输出，格式为[{\"question\":\"xxx\",\"answer\":\"xx\"},....],注意，回答只包含json，不要包含其它内容。如果问题和答案要在特定的条件下才成立，则生成的问题和答案要把条件带上，因为这些问题和答案要脱离这篇文章使用。"},
                {"role": "user", "content": "title: " + data["title"] +"content:\n" + data["content"]},
            ]
            chat = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", tokenize=False
            )
            chats.append(chat)
            if(len(chats) >= batch_size):
                input_ids = tokenizer(chats, return_tensors="pt", padding=True).to(model.device)

                outputs = model.generate(
                    **input_ids,
                    max_new_tokens=8192,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )
                result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                generate_data = generate_data + result
                input_ids.to("cpu")
                torch.cuda.empty_cache()

                chats = []
                # 更新进度条
                pbar.update(batch_size)

drug_dataset = load_dataset("ytcheng/sm_strategy")
drug_dataset
with tqdm(total=total_data) as pbar:
    for i in range(1, 11):
        for data in drug_dataset['train']:
            messages = [
                {"role": "system", "content":"从用户发的文本中，提取出尽可能多的问题和答案，以json格式输出，格式为[{\"question\":\"xxx\",\"answer\":\"xx\"},....],注意，回答只包含json，不要包含其它内容。如果问题和答案要在特定的条件下才成立，则生成的问题和答案要把条件带上，因为这些问题和答案要脱离这篇文章使用。"},
                {"role": "user", "content": "title: " + data["title"] +"content:\n" + data["content"]},
            ]
            chat = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", tokenize=False
            )
            chats.append(chat)
            if(len(chats) >= batch_size):
                input_ids = tokenizer(chats, return_tensors="pt", padding=True).to(model.device)

                outputs = model.generate(
                    **input_ids,
                    max_new_tokens=8192,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )
                result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                generate_data = generate_data + result
                input_ids.to("cpu")
                torch.cuda.empty_cache()

                chats = []
                # 更新进度条
                pbar.update(batch_size)

parseData  = []
for data in generate_data:
    start_index = data.find("```json\n[")+len("```json\n")
    # 找到JSON部分的结束位置
    end_index = data.rfind("]") + 1
    # 提取JSON部分
    json_text = data[start_index:end_index]
    print("json_text:")
    print(json_text)
    
    # content = data["content"].replace("```json", "")
    # content = content.replace("```", "")
    # print(json_text)
    try:
        content = json.loads(json_text)
        # break
        for item in content:
            parseData.append(item)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        print("Skipping this JSON string.")
    # print(content)
    # parseData.append(content)
# print(parseData)
len(parseData)


generate_datasets = Dataset.from_list(parseData)
datasets_formatted_data = DatasetDict({"train":  generate_datasets})
datasets_formatted_data.push_to_hub("ytcheng/sm_question1")