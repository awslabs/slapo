import transformers.utils.fx as fx
from transformers import BertLMHeadModel, BertConfig
import ms

bert = BertLMHeadModel(BertConfig(is_decoder=True))
bert.eval()
gm = fx.symbolic_trace(bert)
sch = ms.create_schedule(gm)
print(sch.forward_ops)
print(sch.modules)