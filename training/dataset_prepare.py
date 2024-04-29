from datasets import load_dataset

# Convert dataset to OAI messages
system_message = """You are Llama, an AI assistant created by Philipp to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""

def create_conversation(sample):
    sample["messages"] = "title:" + sample["title"] + "\n\ncontent: " + sample["content"]
    return sample

# Load dataset from the hub
dataset = load_dataset("ytcheng/sm_article")
dataset = dataset.filter(lambda x: x["content"]!="")

# Add system message to each conversation
columns_to_remove = list(dataset["train"].features)
# columns_to_remove.remove("title")
# columns_to_remove.remove("content")
dataset = dataset.map(create_conversation, remove_columns=columns_to_remove,batched=False)
dataset = dataset["train"].train_test_split(test_size=0.05)

# Filter out conversations which are corrupted with wrong turns, keep which have even number of turns after adding system message
# dataset["train"] = dataset["train"].filter(lambda x: len(x["messages"][1:]) % 2 == 0)
# dataset["test"] = dataset["test"].filter(lambda x: len(x["messages"][1:]) % 2 == 0)

# save datasets to disk
dataset["train"].to_json("./sm/train_dataset.json", orient="records", force_ascii=False)
dataset["test"].to_json("./sm/test_dataset.json", orient="records", force_ascii=False)

