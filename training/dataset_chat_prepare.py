from datasets import load_dataset

# Convert dataset to OAI messages
system_message = """你是蜀门游戏助手，你需要根据回答用户的问题"""

def create_conversation(sample):
    sample["messages"] = [{"role": "system", "content": system_message}, {"role":"user", "content": sample["question"]}, {"role":"assistant", "content":sample["answer"]}]
    return sample

# Load dataset from the hub
dataset = load_dataset("ytcheng/sm_question")

# Add system message to each conversation
columns_to_remove = list(dataset["train"].features)

dataset = dataset.map(create_conversation, remove_columns=columns_to_remove,batched=False)
dataset = dataset["train"].train_test_split(test_size=0.2)

# Filter out conversations which are corrupted with wrong turns, keep which have even number of turns after adding system message
# dataset["train"] = dataset["train"].filter(lambda x: len(x["messages"][1:]) % 2 == 0)
# dataset["test"] = dataset["test"].filter(lambda x: len(x["messages"][1:]) % 2 == 0)

# save datasets to disk
dataset["train"].to_json("./sm_chat/train_dataset.json", orient="records", force_ascii=False)
dataset["test"].to_json("./sm_chat/test_dataset.json", orient="records", force_ascii=False)
