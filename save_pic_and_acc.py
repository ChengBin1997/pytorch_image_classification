from utils import save_pic_and_acc
import json
import os

checkfilename = "log.json"

function = save_pic_and_acc


def eachFile(filepath):
    pathDir = os.listdir(filepath)      #获取当前路径下的文件名，返回List
    for s in pathDir:
        newDir=os.path.join(filepath,s)     #将文件命加入到当前文件路径后面
        if os.path.isfile(newDir) :         #如果是文件
            if checkfilename in newDir:     #判断是否是txt
                function(filepath)
        else:
            eachFile(newDir)                #如果不是文件，递归这个文件夹的路径

eachFile("./")





