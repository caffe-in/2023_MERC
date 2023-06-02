from typing import Any
import torch
import transformers
import pytorch_lightning as pl
from torch.utils.data import DataLoader,Dataset
from pytorch_lightning import LightningModule
from transformers import BertTokenizer,BertModel
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from qdls.object import print_config
import fire
import numpy as np
from data_utility import PerDataMoudle
from sklearn.metrics import f1_score
import evaluate
import datetime
import wandb



dd,hh,mm = datetime.datetime.now().day,datetime.datetime.now().hour,datetime.datetime.now().minute
class EmoSimpleClassifer(pl.LightningModule):
    def __init__(self, config) :
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.l1 = torch.nn.Linear(2048,1024)
        self.l2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(1024,7)
        # self.l4 = torch.nn.ReLU()
        # self.l5 = torch.nn.Linear(512,7)
        self.test_step_outputs = []
        self.test_labels = []
        self.test_predict = []
    def forward(self, emb) :
        outputs = self.l1(emb)
        outputs = self.l2(outputs)
        outputs = self.l3(outputs)
        return outputs
    def training_step(self,batch,batch_idx):
        output = self(batch['emb'])
        label = batch['label']
        
        loss = self.loss_fn(output,label)
        self.log("train_loss",loss)
        return loss
    def validation_step(self,batch,batch_idx):

        output = self(batch['emb'])
        label = batch['label']
        
        loss = self.loss_fn(output,label)
        wandb.log({"val_loss":loss})
        self.log("val_loss",loss,prog_bar=True)
        return loss
    def test_step(self,batch,batch_idx):        
        output = self(batch['emb'])
        label = batch['label']
        
        loss = self.loss_fn(output,label)
        
        self.test_labels.append(label)
        self.test_predict.append(output)
        
        # cal marco_f1 and acc by caffe
        # macro_f1 = self.macro_f1(output,label)
        # acc = self.acc_fn(output,label)
        # self.log("test_macro_f1",macro_f1,prog_bar=True,logger=True)
        # self.test_step_outputs.append([acc,macro_f1])
        
        # cal marco_f1 and acc by huggingface
        
    def on_test_epoch_end(self) -> None:
        macro_f1_metric = evaluate.load("f1")
        acc_metric = evaluate.load("accuracy")
        
        predictions = torch.cat(self.test_predict,dim=0)
        labels = torch.cat(self.test_labels,dim=0)
        predictions = torch.argmax(predictions,dim=1)
        macro_f1 = macro_f1_metric.compute(predictions=predictions,references=labels,labels=[0,1,2,3,4,5,6],average="macro")
        acc = acc_metric.compute(predictions=predictions,references=labels)
        print("the acc for test is {}, the macro for test is {}".format(acc,macro_f1))
        self.test_step_outputs.clear()
        self.test_predict.clear()
        self.test_labels.clear()
        
        
        # mean_all = torch.mean(torch.tensor(self.test_step_outputs,dtype=torch.float),dim=0)
        # print("the acc for test is {}, the macro for test is {}".format(mean_all[0],mean_all[1]))
        # self.test_step_outputs.clear()
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(params=self.parameters(),lr=self.config.learning_rate)
        return optimizer
    def loss_fn(self,outputs,targets):
        return torch.nn.CrossEntropyLoss()(outputs,targets)
    def macro_f1(self,outputs,targets):
        # a function to calculate the macro f1 score
        outputs = torch.argmax(outputs,dim=1)
        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()
        unique_labels = np.unique(targets)
        n_classes = len(unique_labels)
        f1_scores = []
        for label in unique_labels:
            y_true_label = (targets == label).astype(int)
            y_pred_label = (outputs == label).astype(int)
            f1_scores.append(f1_score(y_true_label, y_pred_label))
        macro_f1 = np.mean(f1_scores)
        return macro_f1
    def acc_fn(self,outputs,targets):
        length = outputs.size()[0]
        acc=0
        for idx in range(length):
            output = outputs[idx]
            target = targets[idx]
            # label = 0 if output[0]>output[1] else 1
            # target = 0 if target[0]>target[1] else 1
            flag = torch.argmax(output)==target
            if flag:acc+=1
        return acc/length
class EMOClassifer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = BertModel.from_pretrained(config.pretrained_path)
        self.l1 = torch.nn.Dropout(0.3)
        self.l2 = torch.nn.Linear(768,7)
        self.test_step_outputs = []
        self.test_labels = []
        self.test_predict = []
        
    def forward(self, input_ids,attention_mask,token_type_ids):
        _,outputs= self.model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,return_dict=False)
        outputs = self.l1(outputs)
        outputs = self.l2(outputs)
        return outputs
    def training_step(self,batch,batch_idx):
        output = self(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],token_type_ids=batch['token_type_ids'])
        label = batch['label']
        
        loss = self.loss_fn(output,label)
        self.log("train_loss",loss)
        return loss
    def validation_step(self,batch,batch_idx):
        output = self(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],token_type_ids=batch['token_type_ids'])
        label = batch['label']
        
        loss = self.loss_fn(output,label)
        wandb.log({"val_loss":loss})
        self.log("val_loss",loss,prog_bar=True)
        return loss
    def test_step(self,batch,batch_idx):        
        output = self(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],token_type_ids=batch['token_type_ids'])
        label = batch['label']
        
        loss = self.loss_fn(output,label)
        self.test_labels.append(label)
        self.test_predict.append(output)
        # macro_f1 = self.macro_f1(output,label)
        # acc = self.acc_fn(output,label)
        # self.log("test_macro_f1",macro_f1,prog_bar=True,logger=True)
        # self.test_step_outputs.append([acc,macro_f1])
    def predict_step(self, batch: Any, batch_idx: int):
         output = self(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],token_type_ids=batch['token_type_ids'])
         
         return output
    def on_test_epoch_end(self) -> None:
        macro_f1_metric = evaluate.load("f1")
        acc_metric = evaluate.load("accuracy")
        
        predictions = torch.cat(self.test_predict,dim=0)
        labels = torch.cat(self.test_labels,dim=0)
        predictions = torch.argmax(predictions,dim=1)
        macro_f1 = macro_f1_metric.compute(predictions=predictions,references=labels,labels=[0,1,2,3,4,5,6],average="macro")
        acc = acc_metric.compute(predictions=predictions,references=labels)
        print("the acc for test is {}, the macro for test is {}".format(acc,macro_f1))
        self.test_step_outputs.clear()
        self.test_predict.clear()
        self.test_labels.clear()
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(params=self.parameters(),lr=1e-05)
        return optimizer
    def loss_fn(self,outputs,targets):
        return torch.nn.CrossEntropyLoss()(outputs,targets)
    def macro_f1(self,outputs,targets):
        # a function to calculate the macro f1 score
        outputs = torch.argmax(outputs,dim=1)
        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()
        unique_labels = np.unique(targets)
        n_classes = len(unique_labels)
        f1_scores = []
        for label in unique_labels:
            y_true_label = (targets == label).astype(int)
            y_pred_label = (outputs == label).astype(int)
            f1_scores.append(f1_score(y_true_label, y_pred_label))
        macro_f1 = np.mean(f1_scores)
        return macro_f1
    def acc_fn(self,outputs,targets):
        length = outputs.size()[0]
        acc=0
        for idx in range(length):
            output = outputs[idx]
            target = targets[idx]
            # label = 0 if output[0]>output[1] else 1
            # target = 0 if target[0]>target[1] else 1
            flag = torch.argmax(output)==target
            if flag:acc+=1
        return acc/length


checkpoint_callbacks = callbacks.ModelCheckpoint(
    monitor="val_loss",
    filename="sample-classifier-{epoch:02d}-{val_loss:02f}",
    save_last=False,
    save_top_k=2,
    mode="min",
)


def train_model(config):
    wandb.init(
    project="MERC",
    name=config.version+"-"+str(dd)+"-"+str(hh)+"-"+str(mm),
    config={
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "max_epochs": config.max_epochs,
        "Image_bind":config.Image_bind,
    }
    )

    dm = PerDataMoudle(config)
    dm.setup("fit")
    if config.Image_bind:
        model = EmoSimpleClassifer(config)
    else:
        model = EMOClassifer(config)
    
    logger = TensorBoardLogger(save_dir="..",version=config.version+"-"+str(dd)+"-"+str(hh)+"-"+str(mm))
    trainer = pl.Trainer(
        accelerator='gpu',               
        devices=config.devices,                  
        logger=logger,
        callbacks=[checkpoint_callbacks],
        max_epochs=config.max_epochs,

    )

    trainer.fit(model, datamodule=dm)
def test_model(config):
    
    dm = PerDataMoudle(config)
    dm.setup("test")
    if config.Image_bind:
        model = EmoSimpleClassifer.load_from_checkpoint("/home/zhangqi/project/2023_MERC/lightning_logs/train-image_bind-2-18-58/checkpoints/sample-classifier-epoch=03-val_loss=1.468073.ckpt",config=config)
    else:
        model = EMOClassifer.load_from_checkpoint("/home/zhangqi/project/2023_MERC/lightning_logs/train-image_bind-2-19-40/checkpoints/sample-classifier-epoch=02-val_loss=1.166852.ckpt",config=config)
    logger = TensorBoardLogger(save_dir="..",version=config.version)
    trainer = pl.Trainer(
        accelerator='gpu',               
        devices=config.devices,                  
        logger=logger,
        callbacks=[checkpoint_callbacks],
        max_epochs=config.max_epochs,

    )
    trainer.test(model,dm)
    
def main(config_file=None, version='default_version'):
    if config_file is None:
        raise Exception(f"must specify a configuration file to start!")
    
    config = OmegaConf.load(config_file)
    config.version = version 
    print_config(config)

    if config.mode == 'train':
        train_model(config)
    elif config.mode == "test":
        test_model(config)
    else:
        raise Exception(f"mode `{config.mode}` is not implemented yet!")
 
if __name__=="__main__":
    fire.Fire(main)