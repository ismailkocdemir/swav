# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import math
import os
import shutil
import time
from logging import getLogger


import numpy as np
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.tensorboard import SummaryWriter
#import apex
#from apex.parallel.LARC import LARC

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)
from src.multicropdataset import MultiCropDataset
import src.resnet as resnet_models

logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of STN_VAE_SwAV")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/HDD/DATASETS/",
                    help="path to dataset repository")

#########################
## STN_VAE_swav specific params #
#########################
parser.add_argument("--encoder_hidden_size", default=512, type=int,
                    help="size of hidden layer in the encoder/projection head")
parser.add_argument("--stn_latent_size", default=64, type=int,
                    help="size of the latent dimension in the Spatial Transformer")
parser.add_argument("--vae_latent_size", default=128, type=int,
                    help="size of the latent dimension in the VAE")
parser.add_argument('--penalize_view_similarity', type=bool_flag, default=True, 
                    help='if set, augmented views are penalizied if they are too similar')

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=32, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=0.01, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--wd", default=1e-4, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")


#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="stn_resnet18_vae", type=str, help="convnet architecture")
parser.add_argument("--workers", default=4, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=False,
                    help="whether to train with mixed precision or not")
parser.add_argument("--dump_path", type=str, default="./experiments",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")


def main():
    global args
    args = parser.parse_args()
    #init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build data
    train_dataset = torchvision.datasets.STL10(
        args.data_path,
        split='unlabeled',
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ]),
        download=False
    )
    #sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = resnet_models.__dict__[args.arch](
        normalize=True,
        small_image=True,
        input_shape=[3,96,96],
        stn_latent_size=args.stn_latent_size,
        encoder_hidden_size=args.encoder_hidden_size,
        vae_latent_size=args.vae_latent_size,
        penalize_view_similarity=args.penalize_view_similarity
    )
    
    '''
    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")
    '''

    # copy model to GPU
    model = model.to(device)

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    #optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    '''
    # init mixed precision
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        logger.info("Initializing mixed precision done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )
    '''

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        #amp=apex.amp,
    )
    start_epoch = to_restore["epoch"]

    cudnn.benchmark = True
    summary_writer = SummaryWriter(args.dump_path)
    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)


        scores = train(train_loader, model, optimizer, epoch, lr_schedule, summary_writer)
        training_stats.update(scores)

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            #if args.use_fp16:
            #    save_dict["amp"] = apex.amp.state_dict()
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )


def train(train_loader, model, optimizer, epoch, lr_schedule, summary_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for it, (inputs,targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # ============ multi-view forward passes ... ============
        outputs = model(inputs.to(device))
        loss, loss_vars = model.calculate_loss(outputs)
        
        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        #if args.use_fp16:
        #    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
        #        scaled_loss.backward()
        #else:
        loss.backward()
        optimizer.step()

        # ============ misc ... ============
        losses.update(loss_vars['reconstruction_loss'], inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 50 == 0:
            # update the tensorboard
            summary_writer.add_scalar('lr', lr_schedule[iteration], iteration)
            for k,v in loss_vars.items():
                summary_writer.add_scalar(k, v, iteration)
            visuals_dict = model.get_current_visuals()
            for k,v in visuals_dict.items():
                grid = torchvision.utils.make_grid(v)
                summary_writer.add_image(k, grid, iteration)
            summary_writer.flush()

            # update the logger
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
    return (epoch, losses.avg)



if __name__ == "__main__":
    main()
