import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config.config import config
torch.set_printoptions(profile="full")


class SSTLoss(nn.Module):
    def __init__(self, use_gpu=config['cuda']):
        super(SSTLoss, self).__init__()
        self.use_gpu = use_gpu
        self.max_object = config['max_object']
        self.false_objects_column = None
        self.false_objects_row = None
        self.false_constant = config['false_constant']

    def add_unmatched_dim(self, x):
        if self.false_objects_column is not None and self.false_objects_column.shape[0]!=x.shape[0]:
            self.false_objects_column = None
            self.false_objects_row = None
        if self.false_objects_column is None:
            self.false_objects_column = Variable(
                torch.ones(x.shape[0], x.shape[1], x.shape[2], 1)) * self.false_constant
            if self.use_gpu:
                self.false_objects_column = self.false_objects_column.cuda()
        x = torch.cat([x, self.false_objects_column], 3)

        if self.false_objects_row is None:
            self.false_objects_row = Variable(torch.ones(x.shape[0], x.shape[1], 1, x.shape[3])) * self.false_constant
            if self.use_gpu:
                self.false_objects_row = self.false_objects_row.cuda()
        x = torch.cat([x, self.false_objects_row], 2)
        return x

    def euclidean_dist(self, x, y):
        square_x = x.pow(2)
        square_x = square_x.sum(-1)
        square_y = y.pow(2)
        square_y = square_y.sum(-1)
        ex = square_x.unsqueeze(-1)
        ey = square_y.unsqueeze(-2)
        xy = torch.einsum('bij,bkj->bik', x, y)
        exy = ex - 2 * xy + ey + 1e-10
        dist = exy.sqrt()
        if self.use_gpu:
            dist = dist.unsqueeze(1).cuda()
        else:
            dist = dist.unsqueeze(1)
        dist = self.add_unmatched_dim(dist)
        return dist

    def forward(self, current_feature, next_feature, target, mask0, mask1):
        input = self.euclidean_dist(current_feature, next_feature)
        input = 1 / (input + 0.01)
        mask_pre = mask0[:, :, :]
        mask_next = mask1[:, :, :]

        # TODO: 是否要修改mask的大小？ batch_size 问题需要修改
        mask0 = mask0.unsqueeze(3).repeat(1, 1, 1, self.max_object + 1)
        mask1 = mask1.unsqueeze(2).repeat(1, 1, self.max_object + 1, 1)
        mask0 = Variable(mask0.data)
        mask1 = Variable(mask1.data)
        target = Variable(target.byte().data)

        if self.use_gpu:
            mask0 = mask0.cuda()
            mask1 = mask1.cuda()

        mask_region = (mask0 * mask1).float()  # the valid position mask
        mask_region_pre = mask_region.clone()  # note: should use clone (fix this bug)
        mask_region_pre[:, :, self.max_object, :] = 0
        mask_region_next = mask_region.clone()  # note: should use clone (fix this bug)
        mask_region_next[:, :, :, self.max_object] = 0
        mask_region_union = mask_region_pre * mask_region_next

        input_pre = nn.Softmax(dim=3)(mask_region_pre * input)
        input_next = nn.Softmax(dim=2)(mask_region_next * input)
        input_all = input_pre.clone()
        input_all[:, :, :self.max_object, :self.max_object] = torch.max(input_pre, input_next)[:, :, :self.max_object,
                                                              :self.max_object]
        # input_all[:, :, :self.max_object, :self.max_object] = ((input_pre + input_next)/2.0)[:, :,
        # :self.max_object, :self.max_object]
        target = target.float()
        target_pre = mask_region_pre * target
        target_next = mask_region_next * target
        target_union = mask_region_union * target
        target_num = target.sum()
        target_num_pre = target_pre.sum()
        target_num_next = target_next.sum()
        target_num_union = target_union.sum()
        # todo: remove the last row negative effect
        if int(target_num_pre.item()):
            loss_pre = - (target_pre * torch.log(input_pre)).sum() / target_num_pre
        else:
            loss_pre = - (target_pre * torch.log(input_pre)).sum()
        if int(target_num_next.item()):
            loss_next = - (target_next * torch.log(input_next)).sum() / target_num_next
        else:
            loss_next = - (target_next * torch.log(input_next)).sum()
        if int(target_num_pre.item()) and int(target_num_next.item()):
            loss = -(target_pre * torch.log(input_all)).sum() / target_num_pre
        else:
            loss = -(target_pre * torch.log(input_all)).sum()

        if int(target_num_union.item()):
            loss_similarity = (target_union * (torch.abs((1 - input_pre) - (1 - input_next)))).sum() / target_num
        else:
            loss_similarity = (target_union * (torch.abs((1 - input_pre) - (1 - input_next)))).sum()

        _, indexes_ = target_pre.max(3)
        indexes_ = indexes_[:, :, :-1]
        _, indexes_pre = input_all.max(3)
        indexes_pre = indexes_pre[:, :, :-1]
        mask_pre_num = mask_pre[:, :, :-1].sum().item()
        print('111111111111111111111')
        if mask_pre_num:
            accuracy_pre = (indexes_pre[mask_pre[:, :, :-1]] == indexes_[
                mask_pre[:, :, :-1]]).float().sum() / mask_pre_num
        else:
            accuracy_pre = (indexes_pre[mask_pre[:, :, :-1]] == indexes_[mask_pre[:, :, :-1]]).float().sum() + 1
        print('222222222222222222222')

        _, indexes_ = target_next.max(2)
        indexes_ = indexes_[:, :, :-1]
        _, indexes_next = input_next.max(2)
        indexes_next = indexes_next[:, :, :-1]
        mask_next_num = mask_next[:, :, :-1].sum().item()
        if mask_next_num:
            print('333333333333333333333')
            accuracy_next = (indexes_next[mask_next[:, :, :-1]] == indexes_[
                mask_next[:, :, :-1]]).float().sum() / mask_next_num
            print('444444444444444444444')
        else:
            accuracy_next = (indexes_next[mask_next[:, :, :-1]] == indexes_[mask_next[:, :, :-1]]).float().sum() + 1
        return loss_pre, loss_next, loss_similarity, \
               (loss_pre + loss_next + loss + loss_similarity) / 4.0, accuracy_pre, accuracy_next, (
                       accuracy_pre + accuracy_next) / 2.0, indexes_pre

    def getProperty(self, input, target, mask0, mask1):
        return self.forward(input, target, mask0, mask1)
