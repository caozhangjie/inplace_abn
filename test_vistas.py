import torch
import sys
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Testing script for the Vistas segmentation model")
parser.add_argument("--scales", metavar="LIST", type=str, default="[0.7, 1, 1.2]", help="List of scales")
parser.add_argument("--flip", action="store_true", help="Use horizontal flipping")
parser.add_argument("--fusion-mode", metavar="NAME", type=str, choices=["mean", "voting", "max"], default="mean",
                    help="How to fuse the outputs. Options: 'mean', 'voting', 'max'")
parser.add_argument("--output-mode", metavar="NAME", type=str, choices=["palette", "raw", "prob"],
                    default="final",
                    help="How the output files are formatted."
                         " -- palette: color coded predictions"
                         " -- raw: gray-scale predictions"
                         " -- prob: gray-scale predictions plus probabilities")
parser.add_argument("--rank", metavar="gpu id", type=str, default="0", help="List of GPUs")
parser.add_argument("--world_size", metavar="gpu number", type=str, default="1", help="Number of GPUs")
parser.add_argument("--threshold", metavar="dataset threshold", type=str, default="1000", help="Number of GPUs")
parser.add_argument("snapshot", metavar="SNAPSHOT_FILE", type=str, help="Snapshot file to load")
parser.add_argument("data", metavar="IN_DIR", type=str, help="Path to dataset")
parser.add_argument("output", metavar="OUT_DIR", type=str, help="Path to output folder")

def docstring_hack():
    """
    Multiproc file which will launch a set of processes locally for multi-gpu
    usage: python -m apex.parallel.multiproc main.py ...
    """
    pass

def main():
    # Load configuration
    args = parser.parse_args()

    argslist = list(sys.argv)[1:]
    world_size = torch.cuda.device_count()

    if '--world_size' in argslist:
        world_size = int(argslist[argslist.index('--world_size')+1])
    else:
        argslist.append('--world_size')
        argslist.append(str(world_size))

    workers = []

    gpu_list = args.rank.split(",")
    thres_list = args.threshold.split(",")

    for i in range(world_size):
      if '--rank' in argslist:
        argslist[argslist.index('--rank')+1] = gpu_list[i]
        argslist[argslist.index('--threshold')+1] = thres_list[i]
      stdout = None if i == 0 else open("GPU_"+str(i)+".log", "w")
      print(argslist)
      p = subprocess.Popen([str(sys.executable), 'test_vistas_single_gpu.py']+argslist, stdout=stdout)
      workers.append(p)

    for p in workers:
      p.wait()   


if __name__ == "__main__":
    main()
