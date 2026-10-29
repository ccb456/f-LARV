import os

from transformers import AutoTokenizer, AutoModel

# # 1. 加载 longformer-large-4096 的分词器
# tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096")
#
# # 2. 加载 longformer-large-4096 的模型
# model_plus = AutoModel.from_pretrained("allenai/longformer-large-4096")
#
# # 3. 保存分词器和模型到本地目录
# save_path = "./PLM/longformer-large-4096"  # 保存路径
# tokenizer.save_pretrained(save_path)
# model_plus.save_pretrained(save_path)
#
# print(f"模型和分词器已保存到: {save_path}")

# 1. 加载 roberta-large 的分词器
tokenizer = AutoTokenizer.from_pretrained("roberta-large")

# 2. 加载 roberta-large 的模型
model_plus = AutoModel.from_pretrained("roberta-large")

# 3. 保存分词器和模型到本地目录
save_path = "./PLM/roberta-large"  # 保存路径
if not os.path.exists(save_path):
    os.makedirs(save_path)
tokenizer.save_pretrained(save_path)
model_plus.save_pretrained(save_path)

print(f"模型和分词器已保存到: {save_path}")

# # 1. 加载 bert-base-uncased 的分词器
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# # 2. 加载 bert-base-uncased 的模型
# model = AutoModel.from_pretrained("bert-base-uncased")

# # 3. 保存分词器和模型到本地目录
# save_path = "./PLM/bert-base-uncased"  # 保存路径
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# tokenizer.save_pretrained(save_path)
# model.save_pretrained(save_path)

# print(f"模型和分词器已保存到: {save_path}")