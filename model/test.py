
# 基本パラメータ
model_name = "rinna/japanese-gpt-neox-3.6b"
dataset = "saldra/sakura_japanese_dataset"
is_dataset_local = False
peft_name = "lora-rinna-3.6b-sakura_dataset"
output_dir = "lora-rinna-3.6b-sakura_dataset-results"

# トレーニング用パラメータ
eval_steps = 50 #200
save_steps = 400 #200
logging_steps = 400 #20
max_steps = 400 # dollyだと 4881

# データセットの準備
data = datasets.load_dataset(dataset)
CUTOFF_LEN = 512  # コンテキスト長の上限


model.enable_input_require_grads()
model.gradient_checkpointing_enable()

config = peft.LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.01,
    inference_mode=False,
    task_type=TaskType.CAUSAL_LM,
)

model = peft.get_peft_model(model, config)

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

# プロンプトテンプレートの準備
def generate_prompt(data_point):
    result = f'### 指示:\n{data_point["instruction"]}\n\n### 回答:\n{data_point["output"]}'
    # rinna/japanese-gpt-neox-3.6Bの場合、改行コードを<NL>に変換する必要がある
    result = result.replace('\n', '<NL>')
    return result

VAL_SET_SIZE = 10
# 学習データと検証データの準備
train_val = data["train"].train_test_split(
    test_size=VAL_SET_SIZE, shuffle=True, seed=42
)
train_data = train_val["train"]
train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
val_data = train_val["test"]
val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))


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
        max_steps=max_steps,
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
model.config.use_cache = False
trainer.train()
# LoRAモデルの保存
trainer.model.save_pretrained(peft_name)
print("Done!")