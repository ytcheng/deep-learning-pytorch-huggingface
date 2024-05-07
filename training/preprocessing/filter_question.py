from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch

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

def get_response(output):
    if "assistant\n\n" in output:
        parts = output.split("assistant\n\n", 1)
        return parts[1]
    else:
        return ""

def filter_quesiton(batch):
    chats = []
    for outputs in batch["output"]:
        outputs = map(get_response, outputs)

        messages = [
            {"role": "system", "content":"处理下面几段json，过滤掉question字段不是问句的。如果quesion过长(超过30字)则对quetion进行简化，只保留关键内容。再合并。最后看看是否有意思一样的，如果有意思一样的字数多的。最终输出一段json, 不是要过程，我是要结果。输出结果中只包含json,不要包含其它内容"},
            {"role": "user", "content": "\n".join(outputs)},
        ]
        chat = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", tokenize=False
        )
        chats.append(chat)
    # print(chats)
    input_ids = tokenizer(chats, return_tensors="pt", padding=True).to(model.device)

    outputs = model.generate(
        **input_ids,
        max_new_tokens=4096,
        do_sample=True,
        temperature=0.6,
        top_p=0.8,
    )
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results = map(get_response, results)
    batch["merged"] = results
    torch.cuda.empty_cache()
    get_gpu_memory_usage()

dataset = load_dataset("ytcheng/sm_question1")
dataset = dataset.map(filter_quesiton, batch_size=4, batched=True)

print(dataset)
dataset.push_to_hub("ytcheng/sm_question1")