from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch
from tqdm import tqdm
import json
import re
from datasets import Dataset, DatasetDict
import hashlib

from bs4 import BeautifulSoup

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
    for index, title in enumerate(batch["title"]):

        messages = [
            {"role": "system", "content":"从用户发的文本中，提取出尽可能多的问题和答案，以json格式输出，格式为[{\"question\":\"xxx\",\"answer\":\"xx\"},....],注意，回答只包含json，不要包含其它内容。如果问题和答案要在特定的条件下才成立，则生成的问题和答案要把条件带上，因为这些问题和答案要脱离这篇文章使用。"},
            {"role": "user", "content": "title: " + title +"content:\n" + batch["content"][index]},
        ]
        chat = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", tokenize=False
        )
        chats.append(chat)
        
    input_ids = tokenizer(chats, return_tensors="pt", padding=True).to(model.device)

    outputs = model.generate(
        **input_ids,
        max_new_tokens=4096,
        do_sample=True,
        temperature=0.6,
        top_p=0.8,
    )
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # input_ids.to("cpu")
    torch.cuda.empty_cache()
    get_gpu_memory_usage()
    # results = []
    # for index, sample in enumerate(batch["title"]):
    #     results.append("result:" + str(index))

    if "output" in batch:
        for idx, result in enumerate(results):
            batch["output"][idx].append(result)
    else:
        outputs = []
        for idx, result in enumerate(results):
            output = [result]
            outputs.append(output)
        batch["output"] = outputs
    return batch

pattern = re.compile(r'\[img\][^\[^\]]*\[/img\]')
def remove_html_tags(data):
    soup = BeautifulSoup(data["content"], "html.parser")
    clean_text = soup.get_text()
    clean_text = clean_text.replace('\u3000', '')
    clean_text = clean_text.replace('\xa0','')
    clean_text = re.sub(pattern, ' ', clean_text)

    data["content"] = clean_text
    return data


article_dataset = load_dataset("ytcheng/sm_news")
article_dataset = article_dataset.filter(lambda x: x["content"]!="").filter(lambda x: x["type"]==5)
columns_to_remove = list(article_dataset["train"].features)
columns_to_remove.remove("title")
columns_to_remove.remove("content")
article_dataset = article_dataset.filter(lambda x: x["content"]!="")
article_dataset = article_dataset.map(remove_html_tags, remove_columns=columns_to_remove)
print(article_dataset)
# print(article_dataset["train"][0])

strategy_dataset = load_dataset("ytcheng/sm_strategy")
columns_to_remove = list(strategy_dataset["train"].features)
columns_to_remove.remove("title")
columns_to_remove.remove("content")
strategy_dataset = strategy_dataset.map(remove_html_tags, remove_columns=columns_to_remove)
print(strategy_dataset)
# print(strategy_dataset["train"][0])

dataset = concatenate_datasets([strategy_dataset["train"], article_dataset["train"]])

for i in range(1, 6):
    dataset = dataset.map(generate_question, batch_size=8, batched=True)
    print(dataset[0])
print(dataset)
print(dataset[0])

generate_datasets = Dataset.from_list(dataset)
datasets_formatted_data = DatasetDict({"train":  generate_datasets})
datasets_formatted_data.push_to_hub("ytcheng/sm_question1")