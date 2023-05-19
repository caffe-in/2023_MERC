import json
from collections import OrderedDict
with open("/home/zhangqi/project/2023_MERC/dataset/MERC_Challenge_CCAC2023_train_set/train.json",'r') as f:
    dic = json.load(f,object_pairs_hook=OrderedDict)
print(type(dic))