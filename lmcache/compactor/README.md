## To run lmcache compactor:
### Install lmcache and lmcache-vllm
Clone the repositories:
```git clone -b compaction git@github.com:LMCache/LMCache.git```
```git clone -b compact git@github.com:LMCache/lmcache-vllm.git```

Run ```pip install -e .``` under the following folders:
```LMCache```
```LMCache/csrc```
```lmcache-vllm```

### Run the example
In ```LMCache/examples/compactor/long_gen.py```, run: 
```LMC_COMPACTOR=True VLLM_ATTENTION_BACKEND=XFORMERS python long_gen.py```