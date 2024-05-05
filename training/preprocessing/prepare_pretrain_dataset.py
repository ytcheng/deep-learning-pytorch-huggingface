from datasets import load_dataset, concatenate_datasets, DatasetDict
from bs4 import BeautifulSoup
import re
import time

pattern = re.compile(r'\[img\][^\[^\]]*\[/img\]')
def remove_html_tags(data):
    soup = BeautifulSoup(data["content"], "html.parser")
    clean_text = soup.get_text()
    clean_text = clean_text.replace('\u3000', '')
    clean_text = clean_text.replace('\xa0','')
    clean_text = re.sub(pattern, ' ', clean_text)

    

    data["content"] = clean_text
    return data
def time_format(data):
    timeArray = time.localtime(data["time"])
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    data["time"] = otherStyleTime
    return data

# 处理论坛贴子
forum_dataset = load_dataset(
        "ytcheng/sm_forum"
)
columns_to_remove = list(forum_dataset["train"].features)
columns_to_remove.remove("time")
columns_to_remove.remove("title")
columns_to_remove.remove("content")
# print(columns_to_remove)

forum_dataset = forum_dataset.filter(lambda example: example["status"] == 1)
forum_dataset = forum_dataset.map(remove_html_tags, remove_columns=columns_to_remove)
forum_dataset = forum_dataset.map(time_format)
forum_dataset = forum_dataset.filter(lambda example: len(example["content"]) > 500)
print(forum_dataset)
print(forum_dataset["train"][0])


# 处理攻略站文章
strategy_dataset = load_dataset("ytcheng/sm_strategy")
columns_to_remove = list(strategy_dataset["train"].features)
columns_to_remove.remove("time")
columns_to_remove.remove("title")
columns_to_remove.remove("content")
strategy_dataset = strategy_dataset.map(remove_html_tags, remove_columns=columns_to_remove)
strategy_dataset = strategy_dataset.map(time_format)
print(strategy_dataset)
print(strategy_dataset["train"][0])

# 处理新闻文章
article_dataset = load_dataset("ytcheng/sm_news")
columns_to_remove = list(article_dataset["train"].features)
columns_to_remove.remove("time")
columns_to_remove.remove("title")
columns_to_remove.remove("content")
article_dataset = article_dataset.filter(lambda x: x["content"]!="")
article_dataset = article_dataset.map(remove_html_tags, remove_columns=columns_to_remove)
article_dataset = article_dataset.map(time_format)
print(article_dataset)
print(article_dataset["train"][0])



dataset = concatenate_datasets([forum_dataset["train"], strategy_dataset["train"], article_dataset["train"]])

print(dataset)
print(dataset[0])

dataset = DatasetDict({"train":  dataset})
dataset.push_to_hub("ytcheng/sm_article")
print(dataset)