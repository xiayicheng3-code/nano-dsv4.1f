"""Development microbenchmark; requires the repository test dependencies.

Pass the path to either checkout to compare runtime versions with identical weights.
"""
import sys, json, time, statistics
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'src'),str(root/'tests')]
import torch
from test_vllm_v41_cpu import make_cpu
torch.set_num_threads(1)
model=make_cpu(19)[2]
try:
 from nano_dsv41f.vllm_v41_cpu.session import InferenceSession
except ImportError:
 InferenceSession=None
prompt=(torch.arange(128)[None]%60)+1
model.generate(prompt[:,:8],max_new_tokens=2)
results=[]
for _ in range(3):
 engine=InferenceSession(model,capacity=256) if InferenceSession else model
 start=time.perf_counter();first=engine.generate(prompt,max_new_tokens=16);cold=time.perf_counter()-start
 cold_stats=getattr(engine,'last_stats',{}).copy()
 next_prompt=torch.cat((first,torch.tensor([[2,3,4,5]])),1)
 start=time.perf_counter();second=engine.generate(next_prompt,max_new_tokens=16);warm=time.perf_counter()-start
 results.append(dict(cold_seconds=cold,continuation_seconds=warm,cold_stats=cold_stats,warm_stats=getattr(engine,'last_stats',{})))
print(json.dumps(dict(root=str(root),torch=torch.__version__,threads=1,device='cpu',prompt_tokens=128,new_user_tokens=4,output_tokens=16,repetitions=3,median_cold_seconds=statistics.median(r['cold_seconds'] for r in results),median_continuation_seconds=statistics.median(r['continuation_seconds'] for r in results),results=results)))
