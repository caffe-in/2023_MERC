import torch
from transformers import AutoTokenizer,AutoModel
from torch.utils.data import Dataset,DataLoader
from transformers import GPT2Tokenizer
import os
import json
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, random_split
from omegaconf import OmegaConf
EMO_2_ID={
    "Neutral":0,
    "Surprise":1,
    "Disgust":2,
    "Anger":3,
    "Sad":4,
    "Happy":5,
    "Fear":6
}
class PerDataset(Dataset):
    def __init__(self,config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_path)
        self.max_seq_length = config.max_seq_length
        self.file_path = config.file_path
        with open(self.file_path,'r') as f:
            data_list = f.readlines()
        
        # 将文本转化为 token IDs 并准备好真实标签
        self.data = []
        for sample in data_list:
            sample = sample.strip().split("\t")
            text,label = sample[0],EMO_2_ID[sample[1]] 
            features= self.tokenizer(text, add_special_tokens=True, max_length=self.max_seq_length, truncation=True)
            input_ids = features["input_ids"]
            attention_mask = features["attention_mask"]
            token_type_ids = features["token_type_ids"]
            self.data.append({'input_ids': input_ids,'attention_mask':attention_mask, 'token_type_ids':token_type_ids,'label': label})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
class PerDataMoudle(pl.LightningDataModule):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config
        self.batch_size = self.config.batch_size
        self.dataset = PerDataset(self.config)
        
        
    def setup(self, stage: str) -> None:
        train_size = int(self.config.train_val_test_ratio[0] * len(self.dataset))
        val_size = int(self.config.train_val_test_ratio[1] * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])


    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,num_workers=4,collate_fn=collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size,num_workers=4,collate_fn=collate_fn)
    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=4,collate_fn=collate_fn)

def collate_fn(batch):
    input_ids,attention_mask,token_type_ids,labels = [],[],[],[]
    for example in batch:
        input_ids.append(example["input_ids"])
        attention_mask.append(example["attention_mask"])
        token_type_ids.append(example["token_type_ids"])
        labels.append(example["label"])


    # 将不同长度的输入序列补齐到相同长度
    max_length = max([len(ids) for ids in input_ids])
    input_ids_padded = [ids + [0] * (max_length - len(ids)) for ids in input_ids]
    attention_mask_padded = [ids + [0] * (max_length - len(ids)) for ids in attention_mask]
    token_type_ids_padded = [ids + [0] * (max_length - len(ids)) for ids in token_type_ids]


    # 将列表转换为张量
    input_ids_tensor = torch.tensor(input_ids_padded,dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_mask_padded,dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids_padded,dtype=torch.long)
    labels_tensor = torch.tensor(labels,dtype=torch.long)

    batch = {'input_ids': input_ids_tensor, 'attention_mask':attention_mask_tensor,'token_type_ids':token_type_ids,'label': labels_tensor}
    return batch

def merge_json(folder_path,task="sentiment"):
    all_list = []
    if task=="sentiment":
        file_list = ["positive.json","negative.json"]
    elif task=="topic":
        file_list = ["world.json","sports.json","bussiness.json","science.json"]
    elif task=="all":
        file_list = os.listdir(folder_path)
    for filename in file_list:

        if filename.endswith('.json'):
            path = os.path.join(folder_path, filename)
        
            # 读取JSON文件并将其解析为Python对象
            with open(path, 'r') as f:
                data = f.readlines()
                all_list+=data
    with open(folder_path+f"/{task}.json","w") as f:
        f.writelines(all_list)

def data_trans(file_path:str="../dataset/RUCM3ED/annotation.json",out_path="../dataset/RUCM3ED/annotation.txt",concat_num:int=3):
    with open(file_path,'r') as f:
        dic = json.load(f)
    data_list = []
    emo_set = set()
    count = {}
    flag = True
    for movie in dic.values():
        for movie_slide in movie.values():
            speaker_info = movie_slide["SpeakerInfo"]
            dialog:dict = movie_slide["Dialog"]
            dialog = list(dialog.values())
            if flag:
                print(dialog[:2])
                flag=False
            for index,sentence in enumerate(dialog):
                text = ""
                if index>=concat_num:
                    for i in range(concat_num):
                        text+=dialog[index-i]["Text"]+" "
                else:
                    for i in range(index,-1,-1):
                        text+=dialog[i]["Text"]+" "
                text.strip(" ")
                label = sentence["EmoAnnotation"]["final_main_emo"]
                if label not in count.keys():
                    count[label]=1
                else:
                    count[label]+=1
                data_list.append(text+"\t"+label+"\n")
                emo_set.add(label)
    print(f"the len of data is {len(data_list)}")
    print(emo_set)
    print(count)
    with open(out_path,'w') as f:
        f.writelines(data_list)
            




        ## dic={
        # "shaonianpai": {
        # "shaonianpai_1": {
        #     "SpeakerInfo": {
        #         "A": {
        #             "Name": "王胜男",
        #             "Age": "mid",
        #             "Gender": "female",
        #             "OtherName": []
        #         },
        #         "B": {
        #             "Name": "林大为",
        #             "Age": "mid",
        #             "Gender": "male",
        #             "OtherName": []
        #         }
        #     },
        #     "Dialog": {
        #         "shaonianpai_1_1": {
        #             "StartTime": "00:00:00:01",
        #             "EndTime": "00:00:01:03",
        #             "Text": "钟点工阿姨",
        #             "Speaker": "A",
        #             "EmoAnnotation": {
        #                 "EmoAnnotator1": "Neutral",
        #                 "EmoAnnotator2": "Neutral",
        #                 "EmoAnnotator3": "Neutral",
        #                 "final_mul_emo": "Neutral",
        #                 "final_main_emo": "Neutral"
        #             }
        #         },}

if __name__=="__main__":

    config = OmegaConf.load("/home/zhangqi/project/RUCM3ED/config/RUCM3ED.yaml")
    dm = PerDataMoudle(config)


