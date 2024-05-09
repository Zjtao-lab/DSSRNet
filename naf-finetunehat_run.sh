cd /home/thinkstation03/zjt/NAFNet-ALL/NAFNet-main



# test
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port=4321 basicsr/test.py -opt options/test_finetuneHat/NAFSSR-S_2x_down_reallr_twiceLeft.yml --launcher pytorch


# test debug
# /home/thinkstation03/anaconda3/envs/NAFNet/bin/python basicsr/train.py -opt options/test/NAFSSR/NAFSSR-S_2x.yml


# # x_scale train
# export CUDA_VISIBLE_DEVICES="0,1"
# nohup python -m torch.distributed.launch \
#     --nproc_per_node=2 --master_port=4321 basicsr/train.py \
#     -opt options/train_finetuneHat/NAFSSR-B_x4_down_reallr.yml --launcher pytorch \
#     > ../outprint_simpleffb_B_x2_finetuneHat_down_withReallr_v0.1.txt 2>&1 &




# '''
# nnodes: 表示有多少个节点，可以通俗的理解为有多少台机器，比如nnodes=2，是指有两个节点（即两台机器）参与训练。
# node_rank: 节点的序号，从0开始，比如在A机器上启动时，节点编号是0，node_rank=0；在B机器上启动时，节点编号是1，node_rank=1
# nproc_per_node:一个节点中的进程数量，一般每个进程独占一块GPU，通常也表示为GPU的数量。
# master_addr:master的IP地址，也就是rank=0对应的主机地址。设置该参数为了让其他节点知道主节点的位置，其他节点可以把自己训练的参数传递过去
# master_port:主节点的端口号，用于通信。
# torch.distributed.launch运行代码，每个进程设置5个参数
# （MASTER_ADDR、MASTER_PORT、RANK、LOCAL_RANK和WORLD_RANK）被传入到环境变量中，RANK、LOCAL_RANK和WORLD情况如下：
# RANK： 使用os.environ["RANK"]获取进程的序号，一般1个GPU对应一个进程。它是一个全局的序号，从0开始，最大值为GPU数量-1
# LOCAL_RANK：使用os.environ["LOCAL_RANK"]获取每个进程在所在主机中的序号。从0开始，最大值为当前进程所在主机的GPU的数量-1；
# WORLD_SIZE：使用os.environ["world_size"]获取当前启动的所有的进程的数量（所有机器进程的和），一般world_size = gpus_per_nod
# '''