1.) 
No, but the DWF repo comes with scripts for generating the needed data/samples. The core distinction between the DWF method and others was the emphasis on distinct random inits, which most model zoos don't have or don't make clear (some of the papers are shockingly vague, it's unclear how they even got published).

2.) Nope, never crossed our mind. There might be work in the PEFT/LoRA community on this question, though. 

3.) The DWF-Revisions repo is a clusterfuck from revising for ICLR last year. Its there to point to some scripts that might not have survived the transfer to the official repo. SDAF is an activation space generation project, which will be my MS Thesis. It stands for spec. dec. as fuck (or activation flows...) and is strictly out of scope here. 

4,5,6.) Appendix K was a last minute add, and the method applied is luahgably simple. No CFG, very basic feature injection (not at any intermediate steps/levels), trivial architecture (literally a 3-layer MLP), etc. And the verbage was intentionally choisen to not draw the attention of reviewers, since we didn't want to try to add a whole new contribution during the revisions phase. In fact, if you look at the tables there, the results do seem promising. I think an added regularization from a VAE or sigreg type trick would help the flows live in a smoother space than the raw pca codes. 

7.) yes, the BERT-118Ms were entirely trained from scratch, which took a few days. They were exclusively trained on the Yelp dataset/task and had unique seeds. 

8.) BERT was N=100, and k=99, we always used N=100 and k=N-1 unless we explicity said otherwise. Yes, Transufion was applied to BERT. In fact, we chose BERT instead of GPT2 because BERT has the same transformer encoder blcoks as ViT, which was the 'canonical' application setting of TransFusion. All this to say, BERTs meant we could copy+paste the TransFusion code and not have to worry about any architectural differences. 

9.) Yelp was chosen because it was cheap, well measured, and a regression task (we did an experiment on house pricing regression as well, but it didn't add any content so we didnt include it). The regression/NLP is important bc it covers two bases. 

10.) No fucking idea man, it's a mess. But I'm trying to get him re-engaged. 

11.) I don't think so. He's been more interested in building richer architecture embeddings based on GNN encodings of the model compute graph/block diagrams. 

12.) Yeah it's a mess but would be joint if it works. 

13, 14.) I only recently reviewed it, but I doubt the content changed much. The primary feedback was about structure and some hypernetwrok stuff. Out of scope

15.) DWF showed that as Flow model capacity increases, the benefit of canonicalization vanishes. Canonicalization is slow, expensive, difficult, and doesnt add much when you could get better results by just scaling the flows. 

16.) I think my concern is more aligned with the first statement - that the intermediate models might be shit, and since we want to generate useful models, they would do more harm than good.
