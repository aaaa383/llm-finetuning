import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from colorama import Fore, Back, Style, init
import time
import json
import datasets
from datasets import load_dataset
from InstructDataset import instructDataset
from transformers import DataCollatorForLanguageModeling
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
import os
from peft.utils.config import TaskType
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import peft
import transformers



init(autoreset=True)

model_name = "rinna/japanese-gpt-neox-3.6b"
dataset = "ponponnsan/dolly-sakura"
peft_name = "lora-rinna-3.6b"
output_dir = "lora-rinna-3.6b-results"

# トレーニング用パラメータ
eval_steps = 200
save_steps = 200
logging_steps = 20
CUTOFF_LEN = 256  # コンテキスト長　512にする

# load model
model_path = "rinna/japanese-gpt-neox-3.6b"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16,
    device_map='auto',
    # load_in_8bit=True,
)


# トークナイズ
def tokenize(prompt, tokenizer):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
    )
    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
    }


# データセットの準備
data = load_dataset(dataset)

print(data["train"][0])

# プロンプトテンプレートの準備
def generate_prompt(data_point):
    if data_point["input"]:
        result = f"""
            ### 指示:
            {data_point["instruction"]}

            ### 入力:
            {data_point["input"]}

            ### 回答:
            {data_point["output"]}
            """
    else:
        result = f"""
            ### 指示:
            {data_point["instruction"]}

            ### 回答:
            {data_point["output"]}
            """

    # 改行→<NL>
    result = result.replace('\n', '<NL>')
    return result

# プロンプトテンプレートの確認
print(generate_prompt(data["train"]))

VAL_SET_SIZE = 2000

# 学習データと検証データの準備
train_val = data["train"].train_test_split(
    test_size=VAL_SET_SIZE, shuffle=True, seed=42
)
train_data = train_val["train"]
val_data = train_val["test"]
train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))


# LoRAのパラメータ
lora_config = LoraConfig(
    r= 8, 
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# モデルの前処理
model = prepare_model_for_int8_training(model)

# LoRAモデルの準備
model = get_peft_model(model, lora_config)

# 学習可能パラメータの確認
model.print_trainable_parameters()



# トレーナーの準備
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        num_train_epochs=3,
        learning_rate=3e-4,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        output_dir=output_dir,
        report_to="none",
        save_total_limit=3,
        push_to_hub=False,
        auto_find_batch_size=True
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, cache_dir="./")
# tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir='./')

# # load dataset
# dolly_ja = datasets.load_dataset("kunishou/databricks-dolly-15k-ja")

# dolly_ja_train = list(dolly_ja['train'])

# # print(dolly_ja_train[0])

# prompt_dict = {
#     "prompt_input": (
#         "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。"
#         "要求を適切に満たす応答を書きなさい。\n\n"
#         "### 指示:\n{instruction}\n\n### 入力:{input}\n\n### 応答:"
#     ),
#     "prompt_no_input": (
#         "以下は、タスクを説明する指示です。"
#         "要求を適切に満たす応答を書きなさい。\n\n"
#         "### 指示:\n{instruction}\n\n### 応答:"
#     )
# }

# train_dataset = instructDataset(dolly_ja_train, tokenizer, prompt_dict)
# # print(f"train: {train_dataset[1]}")

# # 受け取ったtensorの中のinput_idsというキーをlabelsという名前のキーにそのままコピーします。
# # その後にinput_idsでpaddingトークンを-100に置換する処理をしています。
# # （つまり指示文のところは特に-100で埋めてるわけではないので、指示文の生成から学習することになる）
# collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)



# # LORA
# for param in model.parameters():
#     param.requires_grad = False # モデルをフリーズ
#     if param.ndim == 1:
#         # 安定のためにレイヤーノルムをfp32にキャスト
#         param.data = param.data.to(torch.float32)

# model.gradient_checkpointing_enable()
# # これがないと動かないことがあるので注意
# model.enable_input_require_grads()

# class CastOutputToFloat(nn.Sequential):
#     def forward(self, x): return super().forward(x).to(torch.float32)
# model.embed_out = CastOutputToFloat(model.embed_out)

# model.gpt_neox.layers[0].attention

# # LoRAのconfigを指定
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     # どのレイヤーをLoRA化したいのか
#     target_modules=["query_key_value"],
#     lora_dropout=0.05,
#     bias="none",
#     fan_in_fan_out=False,
#     task_type=TaskType.CAUSAL_LM
# )

# # ベースモデルの一部をLoRAに置き換え
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

# model.gpt_neox.layers[0].attention


# training_args = TrainingArguments(
#         # モデルの保存先
#         output_dir='./instruction',
#         # チェックポイントを何件残すか
#         save_total_limit=1,
#         # 学習中に１GPUに割り振るバッチサイズ
#         per_device_train_batch_size=8,
#         # 学習のエポック数
#         num_train_epochs=1,
#         # Trueだと、Trainerにわたすデータセットのカラム（今回でいえば、titleとcategory_id）のうちモデルのforward関数の引数に存在しないものは自動で削除されます。
#         # 今回の実装方法はcollatorクラスでtokenizerに通してinput_idsとかを取得したいのでFalse
#         remove_unused_columns=False,
#         logging_steps=20,
#         fp16=True,
#         dataloader_num_workers=16,
#         report_to="none",
# )

# trainer = Trainer(
#         model=model,
#         data_collator=collator,
#         args=training_args,
#         train_dataset=train_dataset,
#     )

model.config.use_cache = False
trainer.train()
model.config.use_cache = True

model.save_pretrained('./instructionTune-sakura')
tokenizer.save_pretrained('./instructionTune-sakura')

print("Done!")
