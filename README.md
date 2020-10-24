# Code to test AdaBelief in Transformer on ISWLT14DE-EN

## set up environment
```conda create -f environment.yml -n nlp```

## install adableief
```pip install adabelief-pytorch```

## set up fairseq
```pip install --editable .```

## prepare data
```sh prepare-iwslt14.sh```

## run code
```sh config/adabelief.sh```

# Results
The BLEU score on my local machine (PyTorch 1.1, CUDA 9.0) is roughly:
AdamW: 35.60         RAdam: 35.51     AdaBelief: 35.85
The result could vary with rnadomness, however they are all above 35.

# Incompatibility between fairseq implementation here and new version PyTorch
When I tested AdaBelief in PyTorch 1.4 and PyTorch 1.6, the BLEU score is always below 30. 
Furthremore, the gradient norm in PyTorch 1.1 is always below 1.0, while with higher version PyTorch the grad explodes to 2 or more.

This seems to be a problem of the version incompatibility between ```fairseq``` here (<=0.8) and ```PyTorch```.<br>
The code here works fine with PyTorch 1.1.<br>
When using PyTorch 1.6, AdaBelief (same code as here) works fine with latest ```fairseq``` implementation.<br>
Code for transformer to work with PyTorch 1.1 is at https://github.com/juntang-zhuang/fairseq-adabelief
