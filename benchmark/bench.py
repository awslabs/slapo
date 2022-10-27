import torch

print('Pytorch version\t:', torch.__version__)
print('CUDA version\t:', torch.version.cuda)

for i in range(torch.cuda.device_count()):
    print(f'GPU{i}\t\t:',torch.cuda.get_device_name(i))

import os
import re
import json

import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from transformers import AutoConfig, PretrainedConfig

@dataclass
class Exp:
    name: str           # Experiment name
    model: str          # huggingface model name
    batch_size: int     # batch size per GPU
    seq_len: int = None # input sequence length
        
    ## Improve speed / reduce memory  
    bf16: bool = False  # Faster, less memory. Recommend if GPU supports
    fp16: bool = False  # Faster, less memory, but need to scale loos. 
                        # Recommend if BF16 is not available.
    optim: str = 'adamw_hf'  # Optimization method
    grad_ckpt: bool = False  # save memory with an extra forward
    grad_accum: int = 1      # accumulate gradients for better performance
    steps: int = 20          # number of parameter updates
        
    ## Multi-GPUs
    gpus: str = '0'          # GPUs to use. "0,1" means use GPU 0 and 1
    tensor_para: int = 1     # Tensor parallelism
    deepspeed: bool = False  # if or not use deepspeed
    ds_config: str = ''      # deepspeed config 
        
    def __post_init__(self):         
        model_conf = AutoConfig.from_pretrained(self.model)
        get = lambda *keys: max([getattr(model_conf, k) if hasattr(model_conf, k) else 0 for k in keys])
        self.num_layers = get('num_hidden_layers', 'n_layer')
        self.num_gpus = len(self.gpus.split(','))                      
        self.hidden_size = get('hidden_size', 'n_embd', 'd_model')
        self.vocab_size = get('vocab_size')
        self.num_heads = get('num_attention_heads', 'n_head')
        if self.seq_len is None:
            self.seq_len = get('max_position_embeddings', 'n_ctx')
        n, h, s, v = self.num_layers, self.hidden_size, self.seq_len, self.vocab_size
        att, ffn, embed = 4*h*s**2 + 8*s*h**2, 16*s*h**2, 2*s*h*v
        forward = n*(att+ffn) + embed
        # TFLOPs to train one example
        self.tflops = (4 * forward if self.grad_ckpt else 3 * forward) / 1e12
        if self.deepspeed:            
            self.launcher = 'deepspeed'            
        else:
            self.launcher = f'torchrun --nproc_per_node {self.num_gpus}' 
            
    def print_results(self):
        print('Total samples / second\t: %.1f' % self.samples_per_sec)
        print('Per GPU memory (GB)\t: %.1f'% self.gpu_mem)
        print('Per GPU TFLOPs\t\t: %.1f' % (self.samples_per_sec * self.tflops / self.num_gpus))

def compare(exps, fig_name):
    fig, ax = plt.subplots(ncols=3, figsize=(9,len(exps)/2))
    x = list(range(len(exps)))
    for i, (y, l) in enumerate((
        ([e.samples_per_sec for e in exps], 'Samples / sec'), 
        ([e.samples_per_sec * e.tflops / e.num_gpus for e in exps], 'per GPU TFLOPS'),
        ([e.gpu_mem for e in exps], 'per GPU memory (GB)'))):
        ax[i].barh(x, y, align='center', height=0.6, color=plt.get_cmap('Set1')(x))
        ax[i].invert_yaxis()
        ax[i].set_xlabel(l)
        if i == 0:
            ax[i].set_yticks(x, labels=[e.name for e in exps])
        else:
            ax[i].set_yticklabels([])
    plt.savefig(fig_name,format="png",dpi=200,bbox_inches='tight')
    plt.show()

def hf_bert(exp):
    cmd = f'''export CUDA_VISIBLE_DEVICES={exp.gpus}; \
{exp.launcher} run_mlm.py \
--config_name {exp.model} --tokenizer_name {exp.model} \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --max_seq_length {exp.seq_len} \
--per_device_train_batch_size {exp.batch_size} \
--fp16 {exp.fp16} --bf16 {exp.bf16} \
--optim {exp.optim} --max_steps {exp.steps} \
--gradient_accumulation_steps {exp.grad_accum} \
--gradient_checkpointing {exp.grad_ckpt} \
--output_dir /tmp/bert/ --overwrite_output_dir yes --skip_memory_metrics False'''
    if exp.deepspeed:
        cmd += f' --deepspeed {exp.ds_config}'
    cmd += ' > log.txt 2>&1'
    os.system(cmd)
    return hf_log(exp, 'log.txt')
    
def hf_log(exp, log_filename):
    with open(log_filename) as f:
        lines = f.readlines()
    for l in lines:
        if 'CUDA out of memory' in l:
            print('Out of GPU memory, try a smaller batch size')
            return None
        if '{\'train_runtime' in l:
            metrics = json.loads(l.replace('\'', '\"'))
            exp.gpu_mem = (metrics['init_mem_cpu_peaked_delta'] + \
                    metrics['train_mem_gpu_alloc_delta'] + metrics['train_mem_gpu_peaked_delta']) / 1e9
            exp.samples_per_sec = metrics['train_samples_per_second']
            return exp
    print(f'Failed. Check "{log_filename}" to find error')    
    return None

def megatron_bert(exp):
    cmd = f'''{exp.launcher} ../../Megatron-LM/pretrain_bert.py \
--num-layers {exp.num_layers} --hidden-size {exp.hidden_size} \
--num-attention-heads {exp.num_heads} \
--tensor-model-parallel-size {exp.tensor_para} \
--micro-batch-size {exp.batch_size} \
--seq-length {exp.seq_len} --max-position-embeddings {exp.seq_len} \
--train-iters {exp.steps} \
--data-path bert-sample_text_sentence \
--vocab-file bert-large-uncased-vocab.txt \
--data-impl mmap --lr 0.00015 --log-interval 5'''
    if exp.bf16: cmd += ' --bf16'
    if exp.fp16: cmd += ' --fp16'
    cmd += ' > log.txt 2>&1'
    os.system(cmd)
    return megatron_log(exp, 'log.txt') 
    
def megatron_log(exp, log_filename):
    with open(log_filename) as f:
        text = f.read()
    # Find the last number after the key, returns 0 if not exists
    query = lambda key: float(next(iter(        
        reversed(re.findall(key+': +([\d\.]+)', text))), 0))
    if 'CUDA out of memory' in text:
        print('Out of GPU memory, try a smaller batch size')
        return
    iter_time = query('elapsed time per iteration \(ms\)') 
    if iter_time == 0:
        print(f'Failed. Check "{log_filename}" to find error')
        return
    exp.samples_per_sec = query('global batch size') / iter_time * 1e3
    exp.gpu_mem = query('max allocated')/1e3
    print('Time breakdown\t\t: forward+backward %.2f, communication %.2f, optimizer %.2f' %(
        (query('forward-compute')+query('backward-compute')) / iter_time, 
        query('backward-params-all-reduce') / iter_time, query('optimizer') / iter_time))        
    return exp

# mega_bert = megatron_bert(Exp('Megatron BERT', 'bert-large-uncased', 8, fp16=True))
# mega_bert_2gpu = megatron_bert(Exp('Megatron BERT (2gpu)', 'bert-large-uncased', 8, fp16=True, gpus="0,1"))
# mega_bert_4gpu = megatron_bert(Exp('Megatron BERT (4gpu)', 'bert-large-uncased', 8, fp16=True, gpus="0,1,2,3"))
# mega_bert_8gpu = megatron_bert(Exp('Megatron BERT (8gpu)', 'bert-large-uncased', 7, fp16=True, gpus="0,1,2,3,4,5,6,7"))
# mega_bert_2gpu = megatron_bert(Exp('Megatron BERT (2gpu)', 'bert-large-uncased', 18, fp16=True, gpus="0,1", tensor_para=2))
# mega_bert_4gpu = megatron_bert(Exp('Megatron BERT (2gpu)', 'bert-large-uncased', 32, fp16=True, gpus="0,1,2,3", tensor_para=4))
# mega_bert_8gpu = megatron_bert(Exp('Megatron BERT (8gpu)', 'bert-large-uncased', 46, fp16=True, gpus="0,1,2,3,4,5,6,7", tensor_para=8))

bert_half = hf_bert(Exp('HF 16-bit', 'bert-large-uncased', 4, fp16=True))
bert_half_2gpu = hf_bert(Exp('HF 16-bit (2 GPU)', 'bert-large-uncased', 4, fp16=True, gpus="0,1"))
bert_half_4gpu = hf_bert(Exp('HF 16-bit (4 GPU)', 'bert-large-uncased', 4, fp16=True, gpus="0,1,2,3"))
bert_half_8gpu = hf_bert(Exp('HF 16-bit (8 GPU)', 'bert-large-uncased', 4, fp16=True, gpus="0,1,2,3,4,5,6,7"))

# compare([mega_bert, mega_bert_2gpu, mega_bert_4gpu])
compare([bert_half, bert_half_2gpu, bert_half_4gpu, bert_half_8gpu], "hf-ms")
