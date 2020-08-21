# PaddlePaddle 计算引擎.
import paddle
from paddle import fluid

# 一些常用的科学计算库.
import numpy as np
import matplotlib.pyplot as plt

EMBEDDING_DIM =64  # 词向量维度.
WINDOW_SIZE =5     #滑动窗口大小.
BATCH_SIZE =200    #迭代 batch 大小.
EPOCH_NUM =10      #训练的轮数.
RANDOM_STATE =0    #设置伪随机数种子.

from paddle.dataset import imikolov

word_vocab =imikolov.build_dict()
vocab_size =len(word_vocab)

# 打印 PTB 数据字典的容量大小.
print("imikolov 字典大小为 "+str(vocab_size))

# 类似 Pytorch 的 DataLoader, 用于在训练时做 batch, 很方便.
data_loader = paddle.batch(imikolov.test(word_vocab, WINDOW_SIZE), BATCH_SIZE)

def build_neural_network():
   assert WINDOW_SIZE %2 ==1
   medium_num = WINDOW_SIZE //2

    # 定义输入变量, 是从文本中截取的连续的文本段.
   var_name_list = [str(i) +"-word"for i in range(0, WINDOW_SIZE)]
   word_list = [fluid.layers.data(name=n, shape=[1], dtype="int64")for n in var_name_list]

    # 取中心词作为输入, 而周围上下文作为输出.
   input_word = word_list[medium_num]
   output_context = word_list[:medium_num] + word_list[medium_num +1:]

    # 将输入输出都做词向量表示, 并且将输出拼起来.
   embed_input = fluid.layers.embedding(
        input=input_word, size=[vocab_size, EMBEDDING_DIM],
       dtype="float32",is_sparse=True,param_attr="input_embedding")
   embed_output_list = [fluid.layers.embedding(
        input=w, size=[vocab_size, EMBEDDING_DIM], dtype="float32",
       is_sparse=True,param_attr="output_embedding")for w in output_context]
   concat_output = fluid.layers.concat(input=embed_output_list,axis=1)

    # 用 -log(sigmoid(score)) 作为度量损失函数.
   var_score =fluid.layers.matmul(embed_input, concat_output, transpose_x=True)
   avg_loss =0-fluid.layers.mean(fluid.layers.log(fluid.layers.sigmoid(var_score)))

    # 使用 Adam 优化算法, 并注意需要返回变量定义名.
   fluid.optimizer.AdamOptimizer().minimize(avg_loss)
   return avg_loss, var_name_list


# 确定执行的环境, 如果支持 CUDA 可以调用CUDAPlace 函数.
device_place =fluid.CPUPlace()
executor = fluid.Executor(device_place)

main_program =fluid.default_main_program()
star_program =fluid.default_startup_program()



# 固定伪随机数种子, 一般用于保证论文效果可复现.
main_program.random_seed =RANDOM_STATE
star_program.random_seed =RANDOM_STATE

# 定义模型的架构 (之前定义函数输出) 以及模型的输入.
train_loss, tag_list =build_neural_network()
feed_var_list =[main_program.global_block().var(n) for n in tag_list]
data_feeder =fluid.DataFeeder(feed_list=feed_var_list, place=device_place)

executor.run(star_program)
for epoch_idx in range(EPOCH_NUM):
    total_loss, loss_list =0.0, []
    for batch_data in data_loader():
       total_loss +=float(executor.run(
           main_program, feed=data_feeder.feed(batch_data),
           fetch_list=[train_loss])[0])
       loss_list.append(total_loss)
    print("[迭代轮数{:4d}], 在训练集的损失为{:.6f}".format(epoch_idx, total_loss))
plt.plot(np.array(range(0, len(loss_list))),loss_list)