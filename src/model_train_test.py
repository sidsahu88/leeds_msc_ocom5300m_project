import torch
from torch.optim import Adam
import numpy as np
import time

import capsule_network as caps
import ccm_pruner as ccmp
import utils

torch.autograd.set_detect_anomaly(True)


def test_capsnet(model, criterion, test_loader, logger, device='cpu'):
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            batch_size = labels.size(0)

            images = images.to(device)
            labels = labels.to(device)

            reconstructed_images, preds, _ = model(images)
            
            if reconstructed_images is None:
                loss = criterion(preds, labels)
            else:
                loss = criterion(preds, labels, images, reconstructed_images)

            losses.update(loss.item(), batch_size)

            preds = preds.norm(dim=-1)
            prec1, prec5 = utils.accuracy(preds, labels, topk=(1, 5))
            top1.update(prec1.item(), batch_size)
            top5.update(prec5.item(), batch_size)

    return losses.avg, top1.avg, top5.avg
    

def train_capsnet(n_epochs, model, criterion, train_loader, test_loader, train_dir, logger, alpha=0.01, calc_ccm=False, ccm_alpha=0.01, device='cpu', checkpoint_file=None):
    optimizer = Adam(model.parameters(), weight_decay=2e-7)
    model = model.to(device)
    epoch_losses = np.zeros((2, n_epochs))
    epoch_accuracies = np.zeros((2, n_epochs))
    
    if checkpoint_file is None:
        best_top1 = 0
        best_trained_model_path = ''
        epoch_start = 1
    else:
        model_checkpoint = torch.load(checkpoint_file, map_location = device)
        epoch_start = model_checkpoint['epoch'] + 1
        
        model.load_state_dict(model_checkpoint['model_state_dict'])
        optimizer.load_state_dict(model_checkpoint['optim_state_dict'])
        
        saved_loss = model_checkpoint['epoch_loss']
        if saved_loss.shape[1] < n_epochs:
            epoch_losses[:,:saved_loss.shape[1]] = saved_loss
        else:
            epoch_losses = saved_loss[:, :n_epochs]
        
        saved_accuracy = model_checkpoint['epoch_accuracy']
        if saved_accuracy.shape[1] < n_epochs:
            epoch_accuracies[:,:saved_accuracy.shape[1]] = saved_accuracy
        else:
            epoch_accuracies = saved_accuracy[:, :n_epochs]
        
        best_top1 = model_checkpoint['best_top1']
        best_trained_model_path = model_checkpoint['best_model_path']
    
    for epoch in range(epoch_start, n_epochs+1):
        batch_time = utils.AverageMeter('Time', ':6.3f')
        data_time = utils.AverageMeter('Data', ':6.3f')
        losses = utils.AverageMeter('Loss', ':.4e')
        top1 = utils.AverageMeter('Acc@1', ':6.2f')
        top5 = utils.AverageMeter('Acc@5', ':6.2f')

        if calc_ccm:
            CCM_losses = utils.AverageMeter('CCM_loss', ':.4e')

        model.train()
        end = time.time()
        
        num_iter = len(train_loader)

        for i, (images, labels) in enumerate(train_loader):
            batch_size = labels.size(0)
            
            images = images.to(device)
            labels = labels.to(device)

            data_time.update(time.time() - end)

            optimizer.zero_grad()

            reconstructed_images, preds, layer_feature_maps = model(images)
            
            if reconstructed_images is None:
                loss = criterion(preds, labels)
            else:
                loss = criterion(preds, labels, images, reconstructed_images)

            if calc_ccm:
                total_CCM_loss = 0
                
                for feature_maps in layer_feature_maps:
                    CCM_loss, _ = ccmp.calc_ccm_loss(feature_maps)
                    total_CCM_loss += CCM_loss
                    
                loss = loss - (total_CCM_loss * ccm_alpha)
                CCM_losses.update(total_CCM_loss.item(), batch_size)

            loss.backward()
            optimizer.step()

            losses.update(loss.item(), batch_size)

            preds = preds.norm(dim=-1)
            prec1, prec5 = utils.accuracy(preds, labels, topk=(1, 5))
            top1.update(prec1.item(), batch_size)
            top5.update(prec5.item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % batch_size == 0:
                logger.info('{} Training - Epoch={}, Iteration=({}/{}), Loss={loss.avg:.4f}, Top 1 Acc={top1.avg:.2f}, Top 5 Acc={top5.avg:.2f}'
                    .format(model.name, epoch, i, num_iter, loss=losses, top1=top1, top5=top5))

        val_loss, val_top1, val_top5 = test_capsnet(model, criterion, test_loader, logger, device)
        
        epoch_losses[:, epoch-1] = (losses.avg, val_loss)
        epoch_accuracies[:, epoch-1] = (top1.avg, val_top1)
        
        logger.info('{} Validation - Epoch={}, Loss={:.4f}, Top 1 Acc={:.3f}, Top 5 Acc={:.3f}'.format(model.name, epoch, val_loss, val_top1, val_top5))
        
        # Save model checkpoint
        checkpoint_filename = "Trained_{0}_Epoch{1}_of_{2}.pt".format(model.name, epoch, n_epochs)
        
        # Identify best accuracy to save best model
        if val_top1 > best_top1:
            best_top1 = val_top1
            best_trained_model_path = train_dir+checkpoint_filename
        
        utils.save_checkpoint(train_dir, checkpoint_filename, epoch, model.state_dict(), optimizer.state_dict(), 
                              epoch_losses, epoch_accuracies, best_top1, best_trained_model_path)
        logger.info("Epoch {} complete. Saved Checkpoint: {}".format(epoch, train_dir+checkpoint_filename))
        torch.cuda.empty_cache()

    logger.info("{} best accuracy={:.3f} saved at: {}".format(model.name, best_top1, best_trained_model_path))

    # Saving best trained model
    utils.save_best_model(train_dir, model.name, best_trained_model_path)

    return best_trained_model_path, epoch_losses, epoch_accuracies