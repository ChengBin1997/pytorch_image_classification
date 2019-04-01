# -*- coding: utf-8 -*-
# @Time    : 2019/3/25 21:44

# @Author  : ChengBin

# @Desc : ==============================================

# ======================================================

import torch.nn.functional as F
import torch

class BaseProcessor(object):
    def __init__(self):
        super(BaseProcessor, self).__init__()

    def __call__(self, inputs, outputs, labels, idx):
        return self.class2one_hot(outputs,labels)

    def class2one_hot(self, outputs, labels):
        class_mask = outputs.new_zeros(outputs.size())
        label_ids = labels.view(-1, 1)
        class_mask.scatter_(1, label_ids, 1.)
        return class_mask



class SampleLabelProcessor(BaseProcessor):
    def __init__(self, Num, alpha=0.01):
        super(SampleLabelProcessor, self).__init__()
        self.alpha = alpha
        self.base_p = 0.5
        self.N = Num
        self.soft_label = None
        self.output_record = False
        self.is_record = False
        self.is_first = True

    def __call__(self, inputs, outputs, labels, idx):
        one_hot_label = self.class2one_hot(outputs, labels)
        if self.soft_label is None:
            self.soft_label = inputs.new_zeros((self.N, outputs.size(1)))
        if self.is_first == True:
            self.soft_label[idx, :] = one_hot_label

        self.soft_label[idx, :] = \
            ((1. - self.alpha) * self.soft_label[idx, :] + \
             self.alpha * F.softmax(outputs, dim=1)).detach()
        self.soft_label[idx, :] = \
            (1. - self.base_p) * self.soft_label[idx, :] + \
            self.base_p * one_hot_label
        return self.soft_label[idx, :]

class SampleLabelProcessCriterion:
    def __init__(self,Num, alpha=0.01):
        self.labelprocessor = SampleLabelProcessor(Num,alpha)



class CategoryLabelProcessor():
    def __init__(self, Num, alpha=0.01,p=0.5,is_select=True):
        self.N = Num
        self.number = torch.zeros(Num,1).cuda()
        self.result = torch.zeros(Num, Num).cuda()
        self.emsemble_label = torch.eye(Num).cuda()
        self.alpha=alpha
        self.p=p
        self.is_select=is_select

    def reset(self):
        self.number = torch.zeros(self.N, 1).cuda()
        self.result = torch.zeros(self.N, self.N).cuda()

    def append(self,output,target):
        batch = target.size(0)
        feature_num =self.N
        mask = torch.zeros(batch, self.N).cuda()
        mask = mask.scatter_(1, target.view(batch, 1).cuda(), 1)  # N*k 每一行是一个one-hot向量

        if self.is_select==True:
            _, predicted = torch.max(output.data, 1)
            index = (predicted == target)
            select = index.view(batch, 1).type(torch.cuda.FloatTensor)
            mask = mask.type(torch.cuda.FloatTensor) * select

        mask = mask.view(batch, self.N, 1) # N*k*1 目的是扩成 N*k*s
        soft_logit = torch.nn.functional.softmax(output, dim=1)
        output_ex = soft_logit.view(batch, 1, feature_num) # N*1*s 目的是扩成 N*k*s
        sum = torch.sum(output_ex * mask, dim=0)

        self.result += sum
        self.number += mask.sum(dim=0)

    def update(self):
        index = (self.number != 0)
        index2 = index.view(1, -1).squeeze()
        newlabel = self.p*torch.eye(self.N)[index2,:].cuda() + (1-self.p)* self.result[index2,:]/self.number[index].view(-1,1)
        self.emsemble_label[index2,:] = (1-self.alpha)*self.emsemble_label[index2,:]+self.alpha*newlabel
        return self.emsemble_label

    def set(self,epoch=None, set_type = None):
        if set_type == 'p_discrete1':
            epochlist=[40,70,100,130]
            plist = [0,0.2,0.5,0.8]
            if epoch in epochlist:
                index = epochlist.index(epoch)
                self.p = plist[index]
        elif set_type == 'p_discrete2':
            epochlist=[40,70,100,130]
            plist = [0.8,0.5,0.2,0]
            if epoch in epochlist:
                index = epochlist.index(epoch)
                self.p = plist[index]


class CategoryLabelProcessCriterion():
    def __init__(self, Num, alpha=0.01, p=0.5, is_select=True):
        self.labelprocessor = CategoryLabelProcessor(Num, alpha, p, is_select)

    def __call__(self, preds, targets):
        emsemble_label = self.labelprocessor.emsemble_label
        loss=my_CrossEntrophyloss(preds,emsemble_label[targets,:])
        return loss


def my_CrossEntrophyloss(logit, prob):
    """ Cross-entropy function"""

    soft_logit = F.log_softmax(logit, dim=1)
    prob.type(torch.cuda.FloatTensor)
    Entrophy = prob.mul(soft_logit)
    loss = -1 * torch.sum(Entrophy, 1)
    loss = torch.mean(loss)

    return loss

