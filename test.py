import os
import time

import cv2
import torch
import torch.utils.data as data
import torch.optim as optim
from visdom import Visdom
import numpy as np

from layer.model import EfficientNet
from efficientnet import EfficientNet as EffNet
from data.mot import MOTTrainDataset
from layer.model_loss import SSTLoss
from utils.augmentations import SSJAugmentation, collate_fn
from config.config import config
from torch.autograd import Variable


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = config['learning_rate'] * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train():
    if config['cuda']:
        model = EfficientNet(0).cuda()
    else:
        model = EfficientNet(0)
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                          momentum=config['momentum'],
                          weight_decay=config['weight_decay'])
    criterion = SSTLoss(config['cuda'])
    if 'learning_rate_decay_by_epoch' in config:
        stepvalues = list((config['epoch_size'] * i for i in config['learning_rate_decay_by_epoch']))
        save_weights_iteration = config['save_weight_every_epoch_num'] * config['epoch_size']
    else:
        stepvalues = (90000, 95000)
        save_weights_iteration = 5000

    # data 处理
    dataset = MOTTrainDataset(config['mot_root'], SSJAugmentation(config['sst_dim'], config['mean_pixel']))
    epoch_size = len(dataset)
    step_index = 0
    data_loader = data.DataLoader(dataset, config['batch_size'], collate_fn=collate_fn)
    batch_iterator = None
    current_lr = config['learning_rate']
    model.train()
    viz = Visdom()
    global_step = 0
    if config['tensorboard']:
        from tensorboardX import SummaryWriter

        if not os.path.exists(config['log_folder']):
            os.mkdir(config['log_folder'])
        writer = SummaryWriter(log_dir=config['log_folder'])
    for iteration in range(config['start_iter'], config['iterations']):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
            all_epoch_loss = []

        if iteration in stepvalues:
            step_index += 1
            current_lr = adjust_learning_rate(optimizer, config['gamma'], step_index)
        img_pre, img_next, boxes_pre, boxes_next, labels, valid_pre, valid_next = next(batch_iterator)
        if config['cuda']:
            img_pre = Variable(img_pre.cuda())
            img_next = Variable(img_next.cuda())
            boxes_pre = Variable(boxes_pre.cuda())
            boxes_next = Variable(boxes_next.cuda())
            with torch.no_grad():
                valid_pre = Variable(valid_pre.cuda())
                valid_next = Variable(valid_next.cuda())
                labels = Variable(labels.cuda())

        else:
            img_pre = Variable(img_pre)
            img_next = Variable(img_next)
            boxes_pre = Variable(boxes_pre)
            boxes_next = Variable(boxes_next)
            with torch.no_grad():
                valid_pre = Variable(valid_pre)
                valid_next = Variable(valid_next)
                labels = Variable(labels)
        t0 = time.time()
        current_feature, next_feature = model(img_pre, img_next, boxes_pre, boxes_next)
        optimizer.zero_grad()
        loss_pre, loss_next, loss_similarity, loss, accuracy_pre, accuracy_next, accuracy, predict_indexes = \
            criterion(current_feature, next_feature, labels, valid_pre, valid_next)
        loss.backward()
        optimizer.step()
        t1 = time.time()
        all_epoch_loss += [loss.data.cpu()]
        if iteration % 10 == 0:
            global_step = global_step + 1
            viz.line([loss.item()], [global_step], win='train_loss', update='append')
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ', ' + repr(epoch_size) + ' || epoch: %.4f ' % (
                    iteration / (float)(epoch_size)) + ' || Loss: %.4f ||' % (loss.item()), end=' ')
        if config['tensorboard']:
            if len(all_epoch_loss) > 30:
                writer.add_scalar('data/epoch_loss', float(np.mean(all_epoch_loss)), iteration)
            writer.add_scalar('data/learning_rate', current_lr, iteration)

            writer.add_scalar('loss/loss', loss.data.cpu(), iteration)
            writer.add_scalar('loss/loss_pre', loss_pre.data.cpu(), iteration)
            writer.add_scalar('loss/loss_next', loss_next.data.cpu(), iteration)
            writer.add_scalar('loss/loss_similarity', loss_similarity.data.cpu(), iteration)

            writer.add_scalar('accuracy/accuracy', accuracy.data.cpu(), iteration)
            writer.add_scalar('accuracy/accuracy_pre', accuracy_pre.data.cpu(), iteration)
            writer.add_scalar('accuracy/accuracy_next', accuracy_next.data.cpu(), iteration)

            # add weights
            if iteration % 1000 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), iteration)
        if iteration % save_weights_iteration == 0:
            print('Saving state, iter:', iteration)
            torch.save(model.state_dict(),
                       os.path.join(
                           config['save_folder'],
                           'sst300_0712_' + repr(iteration) + '.pth'))
    torch.save(model.state_dict(), config['save_folder'] + '' + config['version'] + '.pth')


if __name__ == '__main__':
    train()
