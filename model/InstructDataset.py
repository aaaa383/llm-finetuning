# coding: utf-8
import copy
from tqdm import tqdm
from torch.utils.data import Dataset

# 指示文と回答文を繋げた文章をtokenizeしたtesorを返すだけ。（attention_maskもある）
# labelsについてはDataCollatorForLanguageModelingが内部で用意してくれるので、Datasetで保持する必要なし

class instructDataset(Dataset):
    def __init__(self, json_list, tokenizer, prompt_dict):
        self.tokenizer = tokenizer
        self.prompt_dict = prompt_dict
    
        # print(f'PROMPT_DICT: {prompt_dict}')
        example_texts = []
        for j in json_list:
            # open_qaなど文脈情報が必要ない場合はinputカラムがないため、inputカラムありなしでテンプレート文を分けている。
            if 'input' in j:
                source_text = prompt_dict['prompt_input'].format_map(j)
                print("input",source_text)
            else:
                source_text = prompt_dict['prompt_no_input'].format_map(j)
                print("noinput",source_text)
            
            # 指示文と回答文を結合し、文末にEOSトークンを挿入
            example_text = source_text + j['output'] + self.tokenizer.eos_token
            example_texts.append(example_text)
        
        self.features = [
            tokenizer(
                text+self.tokenizer.eos_token, padding=False, truncation=True, max_length=512
            ) for text in tqdm(example_texts)
        ]
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


