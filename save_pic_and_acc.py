from utils import save_pic_and_acc
import json

path ="./result/CIFAR100/resnet-depth-32"


with open(path+'/log.json', 'r') as json_file:
    parm = json.load(json_file)

train_acc=[]
test_acc=[]
train_loss=[]
test_loss=[]


for epoch in parm:
    train_acc.append(epoch['train']['accuracy'])
    test_acc.append(epoch['test']['accuracy'])
    train_loss.append(epoch['train']['loss'])
    test_loss.append(epoch['test']['loss'])

save_pic_and_acc(train_loss,test_loss,train_acc,test_acc,path)
