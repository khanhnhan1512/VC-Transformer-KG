ffngelu < swiglu+ < ffnrelu < swiglu*
after-ln < pre-ln < post-ln < peri-ln 
blip-avg < blip-cls
peri-ln: phải LN với đầu vào của khối Enc/Dec
1 feat < 2 feat 
skip pre-LN at first enc/dec layer
original pe < rope < new pe < new pe + drop

blip < blip2
4 layer < 3 layer < 2 layer
512 < 640 < 768 ***
6 frame < 8 frame < 10 frame 
i3draft < mvitv2
r2l head: 8 < 128
3 feat < 2 feat 
(attn < ffn)

imgcap: [blipbase + all-MiniLM-L6-v2] < [bliplarge + all-roberta-large-v1]

r2l: 2 feat < 3 feat ***
=> 2e2d < 4e4d < 3e3d
=> 3d: 2e[127.1] < 4e[127.6] < 0e[130.0] < 3e[130.4] < 1e[134.4]

ff < no ff
=> no ff: 0e[137.1] > 1e[133.1] 2e[134.6] 3e[131.2] 4e[133.2]
=> 0e: **3d[137.1]** | 1d[134.5] 2d[131.0] 4d[134.0] 5d[134.1]

512: ffn-relu[136.2] < swiglu*[137.1]
640: swiglu*[132.1] < ffn-relu[135.3]
768: swiglu*[132.9]
384: swiglu*[133.0]

=> new enc - 3d: 1ne[133.5] 2ne[133.9] 3ne[128.4]
no-warmup[135.5] < warmup[137.1]

0e3d-swiglu: 
	64h[134.1] 128h[135.6] < 16h[136.7] < 8head[137.1] < **32head[138.6]**

8head: swiglu*[137.1] < geglu[137.6]
geglu: 
	8head[137.6] | 16h[137.1] 32h[137.0] 64h[136.3] 128h[134.4]

cls-use all(1+10 token): 3e3d[132.9] < 1e1d[134.7] < **2enc2dec[140.4]**
2enc2dec: 640[128.2] < 768[134.3] < **512[140.4]**
2enc2dec-512model: 32h[134.5] < 16h[135.4] < **8h[140.4]**
bighead: 32[130.2] < 64[130.9] < 8[131.5] < 16[136.9] < **128[140.4]**

concat-token: 4e4d[127.7] < 3e3d[128.3] < 2enc2dec[135.0]

pad-token: 3e3d[127.1] < 2enc2dec[135.1]
0enc: 1d[133.4] < 2d[134.8] < **3d[138.9]** > 4d[134.1]

new-pad-token: 
	1e1d[133.0] 2e2d[128.2] < **3e3d[133.4]** > 4e4d[129.4] 5e5d[126.4] 6e6d[124.3]
640model:
	2e2d[128.9] 3e3d[129.3]
0enc3dec: 
	512[134.5] 640[137.5] < **768[138.0]** > 1024[134.4]
3dec-768model:						**0e[138.0]** > 1e[137.2] 2e[130.5]
3dec-768model-EncNoAttn:			**1e[136.8]** 2e[133.6] 3e[133.5]
3dec-768model-PerFeatAttn(sameEnc):	1e[133.1] **2e[137.0]** 3e[127.3]
3dec-768model-SplitEnc: 			**1e[135.3]** 2e[133.8] 3e[131.1] 
3dec-512model-SplitEnc:				1e[132.5] **2e[136.9]** 3e[128.2]
	= 3dec-512model-SplitEncMask:	1e[132.5] **2e[136.9]**
3dec-512model-SplitEncMask-ffplus:	1e[132.3] 2e[131.8] 3e[129.1] **0e[134.2]**
3dec-512model-FuseEncDec:			**1d[131.5]** 2d[130.1] 3d[127.6]
3dec-512model-PerFeatCrossAttn:

best-model: 138.6
	+ enc [v]: 138.6
	+ loader [v]: 138.6
	+ pe(max_len) [v]: 138.6
	+ padmask(and) [v]: 138.6
padmask-or: 138.5 (no mask x1,x2,x3 before plus)
padmask-orv2: 134.2 (mask x1,x2,x3 before plus)
padmask-concat: 136.8
30frame:
	+ padmask-and: 133.5
	+ pasmask-orv2: 135.1
20frame:
	+ padmask-and: 135.6
	+ pasmask-orv2: 134.0
padmask-fuse:
	+ 10frame: 128.0
	+ 30frame: 135.0
padmask-fusev2:
	+ 30frame: 135.4
	+ 20frame: 131.9
padmask-fusev2-3enc:
	+ 20frame: 125.1
	+ 30frame: 127.0
	+ 6frame:  125.4
1024model: 134.6
qformer:
	+ 3enc: 124.5
	+ 1enc: 122.5
EachFeatCrossAttn: 
	+ padmask-or: 131.2
	+ padmask-fuse: 133.3
		+ qnorm [x]: 131.4
		+ no-innorm [x]: 129.7
		+ no-outnorm [x]: 131.5
	
	|n_enc|30frame|20frame|
	|---|---|---|
	|3enc|135.3|**136.3**|
	|2enc|131.8|129.4|
	|4enc|131.2|133.1|
	|1enc|132.0|132.2|
	
	+ 30frame-3enc3dec:
		+  8query: 130.7
		+ 10query: 128.3
		+ 12query: 134.7
		+ 14query: 131.2
		+ 16query: 132.1
		+ 18query: 133.6
		+ 20query: 134.9
		+ 22query: 133.5
		+ 24query: **136.6**
		+ 26query: 129.1
		+ 28query: 132.5
		+ 30query: 135.3
		+ 32query: 133.0

0enc-PerFeatCrossAttnInDec
	+ 10frame 			  (VMC): **136.7**
		+ ChangeFeatOrder (VCM): 132.5
		+ ChangeFeatOrder (MVC): 130.0
		+ ChangeFeatOrder (MCV): 132.9
		+ ChangeFeatOrder (CMV): 132.5
		+ ChangeFeatOrder (CVM): 130.5
	+ 24frame: 129.8
		+ ChangeFeatOrder (VCM): 135.6

VMC:	+ Peri-LN  : 130.7
		+ Pre--LN  :  89.0
		+ Pre--LNv2: 125.0

- MViT-v2 norm?
	+ 10frame: NoNorm[137.0] > Norm[132.1]
	+ 20frame: NoNorm[131.3] < Norm[137.9]
		+ Norm-3enc3dec: 128.7
	+ qformer:
		+ 24frame-3enc3dec: 			131.7
		+ 10frame-2enc3dec-20caplen: 	136.6 (epoch 6)
										131.4 (epoch 11)
		+ 10frame-2e3d-20caplen-maskq:	134.0
		+ 10frame-2enc3dec-30caplen: 	136.5
			+ 768model:					128.6
-[NVIDIA A100] qformer10-newfeats-2enc3dec-512model:
	+  8frame: 134.9
	+ 10frame: 134.9
	+ 12frame: 133.8
	+ 14frame: 137.6
	+ 15frame: 139.0
	+ 16frame: 137.9
	+ 18frame: **140.4**
	+ 20frame: 139.5

gpu04: Tesla V100-DGXS-32GB : 133.7
gpu03: NVIDIA A100-PCIE-40GB: 139.0
gpu01: NVIDIA A100-SXM4-40GB: 139.0
=> A100 has the same result

- FuseEnc(2-1): 130.6
- Enc 2-3 : 133.9
- ParaAttn:
	+ 20frame: 134.6
	+ 18frame: 132.3 (20e) < 137.9 (16e)
	+ 16frame: **136.8**
	+ 14frame: 133.4
	+ 12frame: 135.1
	+ 10frame: 135.3
	+  8frame: 134.9
- ReverseOrder:
	+ 18frame: 139.0
- Parallel:
	+ CA: 	  137.0
	+ CA+FFN: 133.4
- Lowrank:
	+ SharedSA:   131.4
	+ NoSharedSA: 135.4
- FFN:
	+ SwiGLU : 131.7
	+ FFNReLU: 
		+ 18frame: **140.4** => Better on other metrics
		+ 20frame: 138.1


# MSRVTT

40keyint-32head:
	+ 10frame: 61.54
	+ 20frame: 61.46
	+ 24frame: 61.97
qformer-10frame-2enc3dec: 62.4
	+ featFFN before CA:  59.2
	+ keep-feat:		  **62.5**
	+ same-Enc:			  60.6
qformer-10frame-4enc3dec: 60.5
-[NVIDIA A100] qformer10-newfeats-2enc3dec-512model:
	+ 20frame: **62.6**
	+ 18frame: 60.7
	+ 16frame: 61.4
	+ 14frame: 60.8
	+ 12frame: 60.3
	+ 10frame: 62.5
	+  8frame: 60.9

# VATEX

- max 10sec per video: 34352
-[NVIDIA A100] qformer10-newfeats-2enc3dec-512model:
	+ 10frame: 73.0 ['Bleu_4': 39.1, 'METEOR': 26.4, 'ROUGE_L': 53.6, 'CIDEr': 73.0]
	+ ReLU: 'B4': 38.5, 'M': 26.4, 'R': 53.4, 'C': 73.8
	+ GELU: 'B4': 37.9, 'M': 26.3, 'R': 53.0, 'C': 72.9

#SBATCH --nodelist=gpu01

#TODO:
[v] Increase number of keyframes in MSRVTT dataset
[x] Use pretrain `norm` from MViTv2

## Detailed Results

| Method | MSVD<br>(B4-M-R-C) | MSRVTT<br>(B4-M-R-C) |
|---|---|---|
|`BTKG`| 55.7 &nbsp; 38.3 &nbsp; 74.7 &nbsp; 104.5 | 42.8 &nbsp; 30.0 &nbsp; 62.4 &nbsp; 55.4|
|TextKG| 60.8 &nbsp; 38.5 &nbsp; 75.1 &nbsp; 105.2 | 46.6 &nbsp; 30.5 &nbsp; 64.8 &nbsp; 60.8|
|IcoCap (ViT-B/16)| 59.1 &nbsp; 39.5 &nbsp; 76.5 &nbsp; 110.3 | 47.0 &nbsp; 31.1 &nbsp; 64.9 &nbsp; 60.2|
|VASTA (Vatex-backbone)| 59.2 &nbsp; 40.7 &nbsp; 76.7 &nbsp; 119.7 | 44.2 &nbsp; 30.2 &nbsp; 62.9 &nbsp; 56.1|
|SwinBERT| 58.2 &nbsp; 41.3 &nbsp; 77.5 &nbsp; 120.6 | 41.9 &nbsp; 29.9 &nbsp; 62.1 &nbsp; 53.8|
|`CoCap (ViT/L14)`| 60.1 &nbsp; 41.4 &nbsp; 78.2 &nbsp; 121.5 | 44.4 &nbsp; 30.3 &nbsp; 63.4 &nbsp; 57.2|
|VNS-GRU| 66.5 &nbsp; 42.1 &nbsp; 79.7 &nbsp; 121.5 | 45.3 &nbsp; 29.9 &nbsp; 63.4 &nbsp; 53.0|
|*UniVL*| | *42.2* &nbsp; *28.8* &nbsp; *61.2* &nbsp; *49.9*|
|*MV-GPT*| | *48.9* &nbsp; *38.7* &nbsp; *64.0* &nbsp; *60.0*|
|STOA-VLP| 64.4 &nbsp; 43.4 &nbsp; **83.9** &nbsp; 131.8 | 45.8 &nbsp; 31.0 &nbsp; **68.4** &nbsp; 60.2|
|RTQ| 66.9 &nbsp; ----- &nbsp; 82.2 &nbsp; 123.4 | **49.6** &nbsp; ---- &nbsp; 66.1 &nbsp; **69.3**|
|Ours| **67.1** &nbsp; **45.6** &nbsp; 83.0 &nbsp; **140.4** | 47.3 &nbsp; **31.5** &nbsp; 65.0 &nbsp; 62.6|

# asdf

| Method | MSVD<br>(B4-M-R-C) | MSRVTT<br>(B4-M-R-C) |
|---|---|---|
|`BTKG`| 55.7 - 38.3 - 74.7 - 104.5 | 42.8 - 30.0 - 62.4 - 55.4|
|TextKG| 60.8 - 38.5 - 75.1 - 105.2 | 46.6 - 30.5 - 64.8 - 60.8|
|IcoCap (ViT-B/16)| 59.1 - 39.5 - 76.5 - 110.3 | 47.0 - 31.1 - 64.9 - 60.2|
|VASTA (Vatex-backbone)| 59.2 - 40.7 - 76.7 - 119.7 | 44.2 - 30.2 - 62.9 - 56.1|
|SwinBERT| 58.2 - 41.3 - 77.5 - 120.6 | 41.9 - 29.9 - 62.1 - 53.8|
|`CoCap (ViT/L14)`| 60.1 - 41.4 - 78.2 - 121.5 | 44.4 - 30.3 - 63.4 - 57.2|
|VNS-GRU| 66.5 - 42.1 - 79.7 - 121.5 | 45.3 - 29.9 - 63.4 - 53.0|
|STOA-VLP| 64.4 - 43.4 - **83.9** - 131.8 | 45.8 - 31.0 - **68.4** - 60.2|
|RTQ| 66.9 - ----- - 82.2 - 123.4 | **49.6** - ---- - 66.1 - **69.3**|
|Ours| **67.1** - **45.6** - 83.0 - **140.4** | 47.3 - **31.5** - 65.0 - 62.6|

# Step to write paper

(-1.) Abstract: 150-250 words -> main contributions of the paper => write last
	+ Research focus
	+ Research methods
	+ Major results
	+ Main conclusion
(1.) Introduction: introduce contribution of the paper
	+ hardest section
		+ motivation -> what is the problem / why we want to solve and how to solve -> highlight contribution -> overall contribution
(2.) Related works: compare our research with other research
	+ what is the related work?
	+ how you differ
(3.) Define problem:
	+ define precise definition (mathemetical)
(4.) Method: (solution)
	+ diagrams
	+ equations
(5.) Evaluation
(6.) Conclusion and future works
	+ summary
	+ experimental results
	+ conclusion
	+ future work
References