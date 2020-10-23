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

# Problem with PyTorch version
When I tested AdaBelief in PyTorch 1.4 and PyTorch 1.6, the BLEU score is always below 30. 
Furthremore, the gradient norm in PyTorch 1.1 is always below 1.0, while with higher version PyTorch the grad explodes to 2 or more.

I'm not sure what causes this, and I'm still investigating it.
My guess is the denominator st = EMA(sqrt( (gt-mt)^2 )) is too small with higher version PyTorch. It could be either the old version has 
random error in gradient estimation, so st is not so small; or the new version rounds grad to similiar values, hence (gt-mt)^2 is close to 0.

If you have any idea or suggestion, that would be really really helpful.
