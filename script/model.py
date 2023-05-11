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
from data_utility import PerDataMoudle
class EMOClassifer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = BertModel.from_pretrained(config.pretrained_path)
        self.l1 = torch.nn.Dropout(0.3)
        self.l2 = torch.nn.Linear(768,7)
        self.test_step_outputs = []
        
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
        self.log("val_loss",loss)
        return loss
    def test_step(self,batch,batch_idx):        
        output = self(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],token_type_ids=batch['token_type_ids'])
        label = batch['label']
        
        loss = self.loss_fn(output,label)
        acc = self.acc_fn(output,label)
        self.test_step_outputs.append(acc)
    def predict_step(self, batch: Any, batch_idx: int):
         output = self(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],token_type_ids=batch['token_type_ids'])
         
         return output
    def on_test_epoch_end(self) -> None:
        print("the acc for test is {}".format(torch.stack(self.test_step_outputs).mean()))
        self.test_step_outputs.clear()
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(params=self.parameters(),lr=1e-05)
        return optimizer
    def loss_fn(self,outputs,targets):
        return torch.nn.CrossEntropyLoss()(outputs,targets)
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
        return torch.tensor([acc/length],dtype=torch.float)
checkpoint_callbacks = callbacks.ModelCheckpoint(
    monitor="val_loss",
    filename="sample-classifier-{epoch:02d}-{val_loss:02f}",
    save_last=False,
    save_top_k=2,
    mode="min"
)


def train_model(config):
    dm = PerDataMoudle(config)
    dm.setup("fit")
    model = EMOClassifer(config)
    
    logger = TensorBoardLogger(save_dir="..",version=config.version)
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
    model = EMOClassifer.load_from_checkpoint("/home/zhangqi/project/RUCM3ED/lightning_logs/train/checkpoints/sample-classifier-epoch=02-val_loss=1.194331.ckpt",config=config)
    logger = TensorBoardLogger(save_dir="..//lightning_logs",version=config.version)
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