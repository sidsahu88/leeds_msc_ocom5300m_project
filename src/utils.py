import os
import sys
import shutil
import time, datetime
import logging
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import torch

import capsule_network as caps


############################################################################
# Source: https://github.com/Bojue-Wang/CCM-LRR/blob/main/CCM-LRR/utils.py #
############################################################################

'''record configurations'''
class record_config():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        self.args = args
        self.result_dir = Path(args.result_dir)

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.result_dir)

        config_dir = self.result_dir / 'config.txt'

        with open(config_dir, 'w') as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')


def get_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    now = datetime.datetime.now().strftime('%Y_%m_%dT%H_%M_%S')
    logger_file = os.path.join(log_dir, 'capsnet_logger_' + now + '.log')

    logger = logging.getLogger('capsnet')
    log_format = '%(asctime)s | %(message)s'
    
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    
    file_handler = logging.FileHandler(logger_file)
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(file_dir, filename, epoch, model_state_dict, optim_state_dict, train_val_loss, train_val_top1_acc, best_top1, best_model_path):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
    checkpoint_filename = os.path.join(file_dir, filename)
    
    torch.save({"epoch":epoch,
                "model_state_dict":model_state_dict,
                "optim_state_dict":optim_state_dict,
                "epoch_loss":train_val_loss,
                "epoch_accuracy":train_val_top1_acc,
                "best_top1":best_top1,
                "best_model_path":best_model_path}, checkpoint_filename)


def save_best_model(file_dir, model_name, trained_model_filename):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
    best_model_filename = os.path.join(file_dir, "Best_{}.pt".format(model_name))
    
    shutil.copyfile(trained_model_filename, best_model_filename)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def progress_bar(current, total, msg=None):
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)

    TOTAL_BAR_LENGTH = 65.
    last_time = time.time()
    begin_time = last_time

    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('utils')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
    

def plot_line_chart(arrs, legends, xlabel, ylabel, title, xticks=[], marker='o', line='-', color_map = plt.cm.tab10, grid=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if len(xticks) == 0:
        xticks = np.arange(1, len(arrs[0])+1)

    colors = color_map(np.linspace(0, 1, len(legends)))

    for i in range(len(arrs)):
        ax.plot(xticks, arrs[i], marker+line, label = legends[i], color=colors[i])

    ax.set_xticks(xticks)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel(xlabel, fontsize = 16)
    ax.set_ylabel(ylabel, fontsize = 16)
    ax.set_title(title, fontsize = 16)
    ax.legend(fontsize = 12)
    if grid:
        plt.grid(color = 'green', linestyle = '--', linewidth = 0.25)
    fig.tight_layout()


def extract_convcaps_activations(model, test_loader, logger, save_file_dir, device='cpu'):
    caps_activations = {}

    def get_activations(layer_name, n_conv_caps):
        def feature_hook(model, input, output):
            output = output.detach().cpu()
            output = output.view(output.size(0), n_conv_caps, output.size(-1), -1)
            fm_mean = output.mean(dim=-1)
            batch_mean = fm_mean.mean(dim=0)
            norm_output = batch_mean.norm(dim=-1)

            if layer_name not in caps_activations.keys():
                caps_activations[layer_name] = norm_output
            else:
                caps_activations[layer_name] += norm_output
        return feature_hook


    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model = model.to(device)
    model.eval()

    iter = 0

    hook_handles = []

    for layer_name, layer in model.named_modules():
        if isinstance(layer, caps.ConvCapsLayer):
            handle = layer.register_forward_hook(get_activations(layer_name.split('.')[-1], layer.n_out_caps))
            hook_handles.append(handle)

    n_iter = len(test_loader)

    with torch.no_grad():
        for images, labels in test_loader:
            iter += 1

            batch_size = labels.size(0)

            images = images.to(device)
            labels = labels.to(device)

            _, preds, _ = model(images)

            preds = preds.norm(dim=-1)
            prec1, prec5 = accuracy(preds, labels, topk=(1, 5))
            top1.update(prec1.item(), batch_size)
            top5.update(prec5.item(), batch_size)

            logger.info('{} - Iterations={}/{}, Batch Size={}, Top 1 Acc={top1.avg:.3f}, Top 5 Acc={top5.avg:.3f}'
                .format(model.name, iter, n_iter, batch_size, top1=top1, top5=top5))

    for item in hook_handles:
        item.remove()

    for key in caps_activations.keys():
        caps_activations[key] = caps_activations[key]/iter

    save_file_path = save_file_dir +'{}_convcaps_activations.pth'.format(model.name)
    torch.save(caps_activations, save_file_path)
    
    logger.info('{} ConvCaps Activations saved to {}'.format(model.name, save_file_path))
    
    return caps_activations


def plot_caps_layers_activations(caps_activations, n_caps, n_caps_layers, text_en=1):
    activation_matrix = []

    for _, layer_activation in caps_activations.items():
        activation_matrix.append(layer_activation.numpy())

    fig, ax = plt.subplots(figsize=(20,10))

    xlabels = []
    ylabels = []

    for i in range(n_caps):
        xlabels.append('Capsule {}'.format(i+1))

    for j in range(n_caps_layers):
        ylabels.append('ConvCaps Layer {}'.format(j+1))   

    ax.matshow(activation_matrix, cmap=plt.cm.Greens)
    
    if text_en:
        for i in range(n_caps_layers):
            for j in range(n_caps):
                caps_act = round(activation_matrix[i][j], 4)
                ax.text(j, i, str(caps_act), va='center', ha='center')

    plt.xticks(np.arange(n_caps), xlabels, rotation=45)
    plt.yticks(np.arange(n_caps_layers), ylabels)

    plt.show()
    
 
def plot_heatmap_intermediate_layers_activations(caps_activations, n_caps_layers):
    activation_matrix = []

    for _, layer_activation in caps_activations.items():
        activation_matrix.append(layer_activation.numpy())

    fig, ax = plt.subplots(figsize=(10,5))

    xlabels = np.arange(1, len(activation_matrix[0])+1)
    ylabels = np.arange(1, len(activation_matrix)+1)

    ax = sns.heatmap(activation_matrix, linewidth=0.5, cmap=plt.cm.Greens, annot=True, vmin=0.01, vmax=0.1)

    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels, rotation=0)

    ax.set_xlabel('Capsule')
    ax.set_ylabel('Intermediate Layer')

    plt.show()
    

def count_dead_capsules(model_activations, threshold):
    dead_caps = 0
    total_caps = 0

    for layer, activations in model_activations.items():
        activations = torch.round(activations, decimals=3)
        dead_caps += torch.sum(activations <= threshold)
        total_caps += activations.shape[0]

    dead_caps = dead_caps.item()

    return dead_caps, total_caps, round((dead_caps*100)/total_caps, 2)