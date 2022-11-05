import torch
import cv2
import numpy as np
import os
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import time
import argparse
from dataloader import TestDataset
from Net import MyModel
import platform

parser = argparse.ArgumentParser(description='SOD Testing')

# batch_size_eval is the number of images on each GPU, if you have 3 GPUs, the real batch size is (3*batch_size_eval).
parser.add_argument('--batch_size_eval', default=24, type=int, help="batch_size of a GPU")

# test_size, we use the medium size (288) of the five training sizes.
parser.add_argument('--test_size', default=[288, 288], type=list, help="[height, width]")

# path to the weights file
parser.add_argument('--load_model_path', default=r"path/to/model_weights.tar", type=str)

# path to the parent directory of dataset
parser.add_argument('--test_dir', default="path/to/dataset/test", type=str)

# name of the dataset
parser.add_argument('--test_datasets', default=['DUT-OMRON', 'DUTS-TE', 'ECSSD', 'HKU-IS', 'PASCAL-S'], type=list)

# path to save the predicted maps
parser.add_argument('--run_type', default="test_results", type=str)

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


def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank
    dist.init_process_group(backend=backend, init_method='tcp://127.0.0.1:12245', world_size=args.nprocs,
                            rank=local_rank)

    model = MyModel()
    if args.load_model_path is not None and args.load_model_path != "":
        checkpoint = torch.load(args.load_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"load checkpoint {args.load_model_path} success")
    else:
        exit("model weights not found!")
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    test_dataset = TestDataset(test_image_dir=args.test_dir, test_image_datasets=args.test_datasets,
                               test_size=args.test_size)
    test_sampler = DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_eval, pin_memory=True,
                                              drop_last=False, shuffle=False, sampler=test_sampler, num_workers=num_workers)

    val_times = [None for _ in range(args.nprocs)]
    if local_rank == 0:
        val_time = time.strftime("%H-%M-%S")
    else:
        val_time = None
    dist.all_gather_object(val_times, val_time)

    prefix = f"val-time-{val_times[0]}"
    start_test(test_loader, model, local_rank, args, prefix)


def start_test(test_loader, model, local_rank, args, sava_folder):
    model.eval()
    sava_folder = os.path.join(args.run_type, sava_folder)
    if local_rank == 0:
        for dataset in args.test_datasets:
            os.makedirs(os.path.join(sava_folder, dataset))
        pbar = tqdm(total=len(test_loader))
    dist.barrier()
    with torch.no_grad():
        for batch, (inputs, sizes, paths) in (enumerate(test_loader)):
            outputs, _, __, __, __ = model(inputs)
            outputs = round_pixel_uint8(outputs * 255)
            batch_size = outputs.size(0)

            for idx in range(batch_size):
                pred = F.interpolate(outputs[idx].unsqueeze(0), sizes[idx].tolist(),
                                     mode="bilinear").squeeze().detach().cpu().numpy()
                pred = np.uint8(pred)
                cv2.imwrite(os.path.join(sava_folder, paths[idx]), pred)
            if local_rank == 0:
                pbar.update(1)
        dist.barrier()
    if local_rank == 0:
        print(f"Finished! Results are saved in {sava_folder}")


if __name__ == '__main__':
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))
