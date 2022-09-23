import time
import transformers.utils.fx as fx
from transformers import BertLMHeadModel, BertConfig
import torch
import ms

device = "cuda:0"

bert = BertLMHeadModel(BertConfig(is_decoder=True)).to(device)
bert.eval()
gm = fx.symbolic_trace(bert)
sch = ms.create_schedule(gm)
print(sch.forward_ops)
# print(sch.modules)
print(bert.config.vocab_size)

bs = 16
seq_length = 512
bert_input_dict = {
    'input_ids': torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(bert.config.vocab_size),
    'labels': torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(bert.config.vocab_size),
    'attention_mask': torch.ones(bs, seq_length, device=device)}

optimizer = torch.optim.SGD(bert.parameters(), lr=0.001)
for i in range(5):
    start_time = time.time()
    output = bert(**bert_input_dict)
    output.loss.backward()
    optimizer.step()
    elapsed_time = time.time() - start_time
    print(f"Finish step {i}, time: {elapsed_time:.10f}s")
