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
import torch.utils.data as data
import cv2
from moviepy.editor import VideoFileClip
from collections import OrderedDict
from tqdm import tqdm
import logging
import numpy as np
import scipy.io.wavfile as wavfile

sample_rate = 16000
duration = 1
logging.getLogger("tqdm").setLevel(logging.ERROR)
EMO_2_ID={
    "Neutral":0,
    "Surprise":1,
    "Disgust":2,
    "Anger":3,
    "Sad":4,
    "Happy":5,
    "Fear":6
}
class PerSimpleDataset(Dataset):
    """
    for simple dataset with image bind
    """
    def __init__(self,config) -> None:
        self.emb_path = config.emb_path
        self.label_path = config.label_path
        with open(self.label_path,'r') as f:
            self.label_list = f.readlines()
            self.label_list = [EMO_2_ID[label.strip()] for label in self.label_list]
        emb_path_list = os.listdir(self.emb_path)
        self.emb_path_list = [os.path.join(self.emb_path,emb_path) for emb_path in emb_path_list]
        self.emb_list = []
        for emb_path in self.emb_path_list:
            print(emb_path)
            emb = torch.load(emb_path)
            self.emb_list.append(emb)
    def __len__(self):
        return len(self.label_list)
    def __getitem__(self, index):
        emb = []
        for i in range(len(self.emb_list)):
            emb.append(self.emb_list[i][index])
        emb = torch.cat(emb,dim=0)
        label = self.label_list[index]
        return {"emb":emb,"label":label}
        
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
        if self.config.Image_bind:
            self.dataset = PerSimpleDataset(self.config)
        else:
            self.dataset = PerDataset(self.config)
        
        
    def setup(self, stage: str) -> None:
        train_size = int(self.config.train_val_test_ratio[0] * len(self.dataset))
        val_size = int(self.config.train_val_test_ratio[1] * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
        print("filter datset start...")
        print(f"the len of dataset before is {len(self.test_dataset)}")
        self.test_dataset = self.filter_dataset(self.test_dataset)
        print("filter datset end...")
        print(f"the len of dataset after is {len(self.test_dataset)}")
    def filter_dataset(self,dataset):
        indices = [i for i in range(len(dataset)) if dataset[i]["label"]!=0]
        filter_dataset = data.Subset(dataset,indices)
        return filter_dataset


    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,num_workers=4,collate_fn=collate_fn if not self.config.Image_bind else collate_fn_simple)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size,num_workers=4,collate_fn=collate_fn if not self.config.Image_bind else collate_fn_simple)
    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=4,collate_fn=collate_fn if not self.config.Image_bind else collate_fn_simple)

def collate_fn_simple(batch):
    emb_list = []
    label_list = []
    for example in batch:
        emb = example["emb"]
        label = example["label"]
        emb_list.append(emb)
        label_list.append(label)
    emb_list = torch.stack(emb_list,dim=0)
    label_list = torch.tensor(label_list,dtype=torch.long)
    batch = {"emb":emb_list,"label":label_list}
    return batch
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

def get_label(anotation_path:str,wirte_path:str):
    with open(anotation_path,'r') as f:
        label_list = f.readlines()
        label_list = [label.strip().split('\t')[-1] for label in label_list]
    with open(wirte_path,'w') as f:
        label_list = [lable+"\n" for lable in label_list]
        f.writelines(label_list)
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
# "anjia": {
#         "anjia_1": {
#             "SpeakerInfo": {
#                 "A": {
#                     "Name": "房似锦",
#                     "Age": "mid",
#                     "Gender": "female",
#                     "OtherName": []
#                 },
#                 "B": {
#                     "Name": "朱闪闪",
#                     "Age": "young",
#                     "Gender": "female",
#                     "OtherName": []
#                 }
#             },
#             "Dialog": {
#                 "anjia_1_1": {
#                     "StartTime": "00:00:00:01",
#                     "EndTime": "00:00:01:04",
#                     "Text": "你先把衣服换了",
#                     "Speaker": "A",
#                     "EmoAnnotation": {
#                         "EmoAnnotator1": "Neutral",
#                         "EmoAnnotator2": "Neutral",
#                         "EmoAnnotator3": "Neutral",
#                         "final_mul_emo": "Neutral",
#                         "final_main_emo": "Neutral"
#                     }
#                 },
def data_trans(file_path:str="../dataset/RUCM3ED/annotation.json",out_path="../dataset/RUCM3ED/",concat_num:int=3):
    with open(file_path,'r') as f:
        dic = json.load(f,object_pairs_hook=OrderedDict)
    print(type(dic))
    
    folder = os.path.dirname(file_path)
    text_out_path = out_path+"text.txt"
    audio_out_path = out_path+"audio.txt"
    picture_out_path = out_path+"picture.txt"
    
    text_list = []
    audio_list = []
    picture_list = []
    emo_set = set()
    count = {}
    flag = True
    for moive_name,movie in dic.items():
        for moive_slide_name,movie_slide in movie.items():
            speaker_info = movie_slide["SpeakerInfo"]
            
            file_path = f"{folder}/{moive_name}/{moive_slide_name}.mp4"
            
            
            dialog:dict = movie_slide["Dialog"]
            sentences_name_list,sentence_list = dialog.keys(),dialog.values()
            sentences_name_list = list(sentences_name_list)
            sentence_list = list(sentence_list)
            if flag:
                print(sentences_name_list[:2])
                print(sentence_list[:2])
                flag=False
            
            for index,sentence in enumerate(sentence_list):
                print("this is sentence:",sentence)
                sentence_name = sentences_name_list[index]
                text = ""
                start_time = sentence["StartTime"]
                end_time = sentence["EndTime"]
                if index>=concat_num:
                    for i in range(concat_num):
                        text+=sentence_list[index-i]["Text"]+" "
                else:
                    for i in range(index,-1,-1):
                        text+=sentence_list[i]["Text"]+" "
                text.strip(" ")
                label = sentence["EmoAnnotation"]["final_main_emo"]
                if label not in count.keys():
                    count[label]=1
                else:
                    count[label]+=1
                text_list.append(text+"\t"+label+"\n")
                audio_list.append(file_path+"\t"+start_time+"\t"+end_time+"\n")
                picture_list.append(file_path+"\t"+start_time+"\n")
                emo_set.add(label)
                
    print(f"the len of data is {len(text_list)}")
    print(emo_set)
    print(count)
    with open(text_out_path,'w') as f1,open(audio_out_path,'w') as f2,open(picture_out_path,'w') as f3:
        f1.writelines(text_list)
        f2.writelines(audio_list)
        f3.writelines(picture_list)
        
            




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

def write_picture(picture_message_path:str="../dataset/RUCM3ED/picture.txt",out_path:str="../dataset/RUCM3ED/picture"):
    with open(picture_message_path,'r') as f:
        picture_message_list = f.readlines()
    picture_message_list = [picture_message.strip().split("\t") for picture_message in picture_message_list]

    
    for index,picture_message in tqdm(enumerate(picture_message_list)):
        snapshot_time_list = picture_message[1].split(":")
        snapshot_time = int(snapshot_time_list[0])*3600+int(snapshot_time_list[1])*60+int(snapshot_time_list[2])+int(snapshot_time_list[3])/1000
        picture = get_picture(picture_message[0],snapshot_time)
        cv2.imwrite(out_path+f"/{index}.jpg",picture)
def write_audio(audio_message_path:str="../dataset/RUCM3ED/audio.txt",out_path:str="../dataset/RUCM3ED/audio"):
    position_bias = 8232
    audio_error_num = 0
    audio_error_list = []
    with open(audio_message_path,'r') as f:
        audio_message_list = f.readlines()[position_bias:]
    audio_message_list = [audio_message.strip().split("\t") for audio_message in audio_message_list]
    for index,audio_message in tqdm(enumerate(audio_message_list)):
        index = index+position_bias
        start_time_list = audio_message[1].split(":")
        end_time_list = audio_message[2].split(":")
        start_time = int(start_time_list[0])*3600+int(start_time_list[1])*60+int(start_time_list[2])+int(start_time_list[3])/1000
        end_time = int(end_time_list[0])*3600+int(end_time_list[1])*60+int(end_time_list[2])+int(end_time_list[3])/1000
        audio = get_audio(audio_message[0],start_time,end_time)
        try:
            audio.write_audiofile(out_path+f"/{index}.wav")
        except (OSError,IndexError):
            samples = np.random.uniform(-1,1,size=(sample_rate*duration,))
            wavfile.write(out_path+f"/{index}.wav",sample_rate,samples)
            audio_error_num+=1
            audio_error_list.append(index)
        if index%100==0:
            print(f"the audio error num is {audio_error_num}")
    print(f"the audio error num is {audio_error_num}")
    print(f"the audio error list is {audio_error_list}")
        
    
def get_picture(video_path,snapshot_time):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, int(snapshot_time*1000))
    success, image = cap.read()
    cap.release()
    if success:
        return image
    else:
        raise Exception("get picture error")
def get_audio(video_path,start_time,end_time):
    video = VideoFileClip(video_path)
    audio = video.subclip(start_time,end_time).audio
    return audio
if __name__=="__main__":

    config = OmegaConf.load("/home/zhangqi/project/2023_MERC/config/RUCM3ED.yaml")
    dm = PerDataMoudle(config)
    # data_trans(file_path="/home/zhangqi/project/2023_MERC/dataset/MERC_Challenge_CCAC2023_train_set/train.json")
    # write_picture()
    #/home/zhangqi/project/2023_MERC/dataset/MERC_Challenge_CCAC2023_train_set/jinhun/jinhun_17.mp4	00:01:10:00	00:01:11:01
    # start_time_list = "00:01:10:00".split(":")
    # start_time = int(start_time_list[0])*3600+int(start_time_list[1])*60+int(start_time_list[2])+int(start_time_list[3])/1000
    # end_time_list = "00:01:11:01".split(":")
    # end_time = int(end_time_list[0])*3600+int(end_time_list[1])*60+int(end_time_list[2])+int(end_time_list[3])/1000
    # print(start_time,end_time)
    # audio = get_audio("/home/zhangqi/project/2023_MERC/dataset/MERC_Challenge_CCAC2023_train_set/jinhun/jinhun_17.mp4",start_time,end_time)
    # audio.write_audiofile("test.wav")
    # write_audio()
    # get_label("/home/zhangqi/project/2023_MERC/dataset/RUCM3ED/annotation.txt","/home/zhangqi/project/2023_MERC/dataset/RUCM3ED/emb/label.txt")
    # ds = PerSimpleDataset(config=OmegaConf.load("/home/zhangqi/project/2023_MERC/config/RUCM3ED.yaml"))
    # print(ds[0])
    
   

