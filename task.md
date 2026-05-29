# This document summarizes the initial task posed to Claude code

### Goal: 
To build a VAE for LLM weights. 

### Method:
An LLM is a stack of transformer decoder blocks. Rather than working in the full rank of the weight space, we can use a block number embeding as a condition to build a VAE that works block-wise. Since not all blocks are the same size, we will need to use the largest block as the VAE input size, and pad the smaller blocks to fit that dimension. These padded portions should be masked out of the reconstruction loss for those blocks so they do not skew training dynamics. We will also need to account for this across model families (ie, llama and gemma). 

So, our data curation will look like: 
LLM(family_a) -> { block_1|block_2|...|block_L}

And the VAE task like: 
Encoder( block_i | block_index | model_family) -> z = (mu, logvar) -> Decoder(z | block_index | model_family)
with standard VAE reconstruction loss dynamics. 

Further subdivision of blocks between attention and other weights might be needed later, but for now let's postpone that option. 

For the moment, I want to consider a small group of models. If they are for some reason unyeildy, you can expand or revise the group ad-hoc. I would suggest the following: 
*   - openai-community/gpt2-medium
*   - HuggingFaceTB/SmolLM2-360M
*   - google/gemma-3-270m
*   - facebook/opt-350m

These should all be roughly similar sizes and have similar basic transformer deocder block stacking. 

To augment the dataset, we can use on the fly noise augmentation of tiny order. This needs to be low relative to the scale of the weights, maybe on the order of 1e-6 at most. This will help smooth learning dynamics, and should have low cost as it can be easily done with Torch on the fly (at training time, but if doing it at data curation time is easier that is also an option). 

### Success: 

You have succeeded at this task when: 
* We have trained a working VAE for this task on the HPC system accessible via your ssh 'explorer' tool. Note that you are NOT natively in this environment. 
* We have evaluated the reconstructed models on a few donwstream measures and recover comparable quality to originals. There are several potential dimensions of this, so start simple and we can expand later.


### Reference:
I will include some references in your prompt, but for quidance on the VAE design and model chunking approach, you can check these repos: 
* https://github.com/ScottBiggs2/DeepWeightFlow-Revisions 
* https://github.com/ScottBiggs2/SDAF 
