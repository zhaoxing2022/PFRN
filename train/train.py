import torch
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import time
import argparse
from dataloader import TrainDataset,collate
from Net import MyModel
import pytorch_ssim,pytorch_iou
import platform


parser = argparse.ArgumentParser(description='SOD Training')
parser.add_argument('--num_epochs', default=80, type=int)

# batch_size_train is the number of images on each GPU, if you have 3 GPUs, the real batch size is (3*batch_size_train).
parser.add_argument('--batch_size_train', default=8, type=int, help="batch_size is the number on a gpu")

parser.add_argument('--train_dir', default="path/to/dataset/train", type=str)
parser.add_argument('--train_datasets', default=['DUTS-TR'], type=list)

# path to save the logs and models
parser.add_argument('--run_type', default="train_results", type=str)

#difference between windows and ubutun
if platform.system().lower() == 'windows':
    backend = 'gloo'
    num_workers = 0
else:
    backend = "nccl"
    num_workers = 6

class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()

        return grad_input
round_pixel_uint8 = Round.apply

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt.item()

def bce_ssim_loss(bce_loss, ssim_loss, iou_loss, pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + ssim_out + iou_out
    return loss

def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank
    dist.init_process_group(backend=backend, init_method='tcp://127.0.0.1:12245', world_size=args.nprocs, rank=local_rank)

    if args.local_rank == 0:
        #path_save_logs
        args.log_loss_file = f"{args.run_type}/log.txt"
        #path_save_models
        args.model_save_dir = f"{args.run_type}/models"
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)

        with open(args.log_loss_file, 'a') as f:
            f.write(f"train_time: {time.strftime('%m-%d-%H-%M-%S')}\n")

    model = MyModel()
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = torch.nn.BCELoss().cuda(local_rank)
    pv_loss = torch.nn.SmoothL1Loss(reduction='sum',beta=0.5).cuda(local_rank)

    ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True).cuda(local_rank)
    iou_loss = pytorch_iou.IOU(size_average=True).cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    resnet_params, other_params = [], []
    for name, param in model.named_parameters():
        if 'resnet' in name:
            resnet_params.append(param)
        else:
            other_params.append(param)
    optimizer = torch.optim.Adam([{'params': resnet_params, 'lr': 1e-5}, {'params': other_params}], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=15, gamma=0.5)

    train_dataset = TrainDataset(train_image_dir=args.train_dir, train_image_datasets=args.train_datasets)
    image_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size_train , pin_memory=True,drop_last=True,sampler=image_sampler, num_workers=num_workers,collate_fn=collate)


    for epoch in range(args.num_epochs):
        image_sampler.set_epoch(epoch)

        if epoch < 20:
            pretain(train_loader, model, criterion,ssim_loss, iou_loss, optimizer, epoch,local_rank,args)
        else:
            finetune(train_loader, model, criterion, pv_loss,ssim_loss,iou_loss,optimizer,epoch,local_rank, args)
            dist.barrier()
            if local_rank == 0:
                with open(args.log_loss_file, 'a') as file_object:
                    print('Epoch{}----learning:{}--{}\n'.format(str(epoch), optimizer.param_groups[0]["lr"],optimizer.param_groups[1]["lr"]))
                    file_object.write('Epoch{}----learning:{}--{}\n'.format(str(epoch), optimizer.param_groups[0]["lr"],optimizer.param_groups[1]["lr"]))
                torch.save({'model_state_dict': model.module.state_dict()}, f"{args.model_save_dir}/epoch{epoch}.tar")
            dist.barrier()
            scheduler.step()
def pretain(train_loader, model, criterion,ssim_loss, iou_loss, optimizer, epoch, local_rank, args):
    loss_mean=0.
    loss_pred_mean = 0.
    loss_128_mean = 0.
    loss_64_mean = 0.
    loss_32_mean = 0.
    loss_16_mean = 0.
    if local_rank == 0:
        pbar = tqdm(total=len(train_loader))
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(local_rank, non_blocking=True)
        labels = labels.to(local_rank, non_blocking=True)

        optimizer.zero_grad()
        outputs_pred, outputs_128, outputs_64, outputs_32, outputs_16 = model(inputs)
        loss_pred = bce_ssim_loss(criterion, ssim_loss, iou_loss, outputs_pred, labels)
        loss_128 = bce_ssim_loss(criterion, ssim_loss, iou_loss, outputs_128, labels)
        loss_64 = bce_ssim_loss(criterion, ssim_loss, iou_loss, outputs_64, labels)
        loss_32 = bce_ssim_loss(criterion, ssim_loss, iou_loss, outputs_32, labels)
        loss_16 = bce_ssim_loss(criterion, ssim_loss, iou_loss, outputs_16, labels)

        loss = loss_pred + loss_128 * 0.5 + loss_64 * 0.4 + loss_32 * 0.3 + loss_16 * 0.2
        loss.backward()
        optimizer.step()

        reduce_loss = reduce_mean(loss, args.nprocs)
        reduce_loss_pred = reduce_mean(loss_pred, args.nprocs)
        reduce_loss_128 = reduce_mean(loss_128, args.nprocs)
        reduce_loss_64 = reduce_mean(loss_64, args.nprocs)
        reduce_loss_32 = reduce_mean(loss_32, args.nprocs)
        reduce_loss_16 = reduce_mean(loss_16, args.nprocs)

        if local_rank == 0:
            pbar.update(1)
            loss_mean += reduce_loss
            loss_pred_mean += reduce_loss_pred
            loss_128_mean += reduce_loss_128
            loss_64_mean += reduce_loss_64
            loss_32_mean += reduce_loss_32
            loss_16_mean += reduce_loss_16

    if local_rank == 0:
        pbar.close()
        loss_mean /= (i + 1)
        loss_pred_mean /= (i + 1)
        loss_128_mean /= (i + 1)
        loss_64_mean /= (i + 1)
        loss_32_mean /= (i + 1)
        loss_16_mean /= (i + 1)
        with open(args.log_loss_file, 'a') as file_object:
            info = f"loss-epoch{epoch}:loss:{loss_mean}, loss_pred:{loss_pred_mean}, loss_128:{loss_128_mean}, loss_64:{loss_64_mean}, loss_32:{loss_32_mean}, loss_16:{loss_16_mean}"
            print(info)
            file_object.write(f"{info}\n")


def finetune(train_loader, model, criterion, pv_loss,ssim_loss, iou_loss, optimizer, epoch, local_rank, args):
    loss_pred_mean = 0.
    loss_pv_mean = 0.
    if local_rank == 0:
        pbar = tqdm(total=len(train_loader))
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(local_rank, non_blocking=True)
        labels = labels.to(local_rank, non_blocking=True)

        optimizer.zero_grad()

        outputs_pred, _, _, _, _ = model(inputs)
        loss_pred = bce_ssim_loss(criterion, ssim_loss, iou_loss, outputs_pred, labels)

        #generating maps into training loss.
        pred_png = round_pixel_uint8(outputs_pred*255)/255
        pred_png_clone = pred_png.clone().detach()
        labels_pv = labels.clone().detach()

        #PV loss don't take the pure pixels(0 or 1) into consideration.
        position_0 = (pred_png_clone == 0.)
        position_1 = (pred_png_clone == 1.)
        labels_pv[position_0]=0
        labels_pv[position_1]=1

        #number of pixels between (0,1), dynamic changing on each different batch.
        num_of_pv_pixels = (torch.numel(pred_png_clone)-position_0.sum()-position_1.sum()).item()
        loss_pv = pv_loss(pred_png,labels_pv)/num_of_pv_pixels


        loss = loss_pred + loss_pv
        loss.backward()

        optimizer.step()

        reduce_loss_pred = reduce_mean(loss_pred, args.nprocs)
        reduce_loss_pv = reduce_mean(loss_pv, args.nprocs)

        if local_rank == 0:
            pbar.update(1)
            loss_pred_mean += reduce_loss_pred
            loss_pv_mean += reduce_loss_pv

    if local_rank == 0:
        pbar.close()
        loss_pred_mean /= (i + 1)
        loss_pv_mean /= (i + 1)
        with open(args.log_loss_file, 'a') as file_object:
            info = f"loss-epoch{epoch}:loss_pv:{loss_pv_mean}, loss_pred:{loss_pred_mean}"
            print(info)
            file_object.write(f"{info}\n")

if __name__ == '__main__':
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))