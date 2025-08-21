Timing

Phase 0: August 5th
Phase 1: August 8th August 15th
Phase 2: August 15th

Notes:
Aim to frontload the programming effort and finish it by the end of next week for Andrew’s scheduling, since he will be doing an AI safety thing in Berkeley in late August.
The UConn semester also begins in late August, so Aidan (and perhaps Kaley?) will begin TA work around then
Aidan is likely to be less active for 10-14 days in early August due to minor surgery on his dominant hand. His plan is to focus on pair programming and other tasks that are light on typing during that period.

Phase 0 - Data preprocessing

Human Annotations: 
Filter the human annotations to exclude trivial maps and retain only high-quality ones. NOTE: We need to set a standard for what defines a high quality map and reference it in the manuscript (e.g. ≥ n nodes)
Parse Argdown format into a structured JSON schema:
ADUs (ID, text span, labels - claim/premise)
Relations (edge label, direction - src/target ADU id)
Metadata (paper identifier, annotator username)
NOTE: See AM Schema for the standardized format.
Manually annotate/infer edge labels of annotator ADU relations. (We had annotators mark implicit/explict claims in Argdown format using +/- instead of using those to label support/attack relations.) NOTE: We could ignore support/attack labelling but that's a common argument map feature I think we should (need to) have.
Stratify the gold-standard set by paper category or map size, then split into training vs. held-out sets (80/20).

Raw Papers:
Run clean-paper.py (may need to be updated) across all Docling .txt/.md paper extractions to normalize Unicode, strip artifacts, etc.

Notes: Our sense of what a high-quality map looks like may change once we start reviewing; by Monday or Tuesday (August 4th or 5th) we should have a grading template and an updated sense of the timeline necessary for reviewing. It should be possible to finish this by Wednesday, but it’s okay if this happens concurrently with Phase 1 and finishes on Friday, August 8th.


Phase 1 - Fit hyperparams against human annotations

Goal: Find, for each candidate language model, the hyperparameters that minimize a multi-feature loss function against the workshop argument map annotations.

Data: 
Input: Papers and, if applicable, prompts.
Output: ADUs and relations (span and edge labels). 
Eval: Valid split of the curated argument maps. (Heldout saved for down the line.)

Cost function: 
Fuzzy-match F1 Scores: Token-overlap F1 score between predicted vs. gold-standard ADU spans.

ADU/R Count: RMSE on the total number of spans and relations (expected = gold-standard, actual = target architecture results). NOTE: we should discuss advantages of RMSE vs. MAE vs. etc.
	
Relation-type F1 Scores: compute F1 separately for support/attack edges (treating each as the positive class against the others), then average to get a macro-F1 score.

Graph Edit Distance: compute attributed GED for the graph structure. See networkx docs for details. Node match function will be crossing the fuzzy F1 threshold. Edge match function will be binary T/F on the support/attack relationship.
	
Loss Function:

Equal weight is thus wi = ¼

Models:
(LLM) unsloth/Meta-Llama-3.1-8B-bnb-4bit
Fine-tune: RIET-lab/AMwithLLMs-Meta-Llama-3.1-8B-Instruct-bnb-4bit (TODO)
Paper: Argument Mining with Fine-Tuned Large Language Models (2025)
(PLM) raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L
Paper: Transformer-Based Models for Automatic Identification of Argument Relations: A Cross-Domain Evaluation (2021)
(PLM) chkla/roberta-argument
Paper: Cross-topic Argument Mining from Heterogeneous Sources (2018)
(PLM + BiLSTM + CRF) ArneBinder/sam-adur-sciarg
Paper: Full-Text Argumentation Mining on Scientific Publications (2022)
(PLM + BiLSTM/CNN + FC) ArneBinder/sam-are-sciarg
Paper: Full-Text Argumentation Mining on Scientific Publications (2022)

Argument Mining Process: AM is split into ADU recognition (ADUR) and argumentative relationship extraction (ARE). In other words, identify ADUs in the text → classify relationships between them.

Pipelines
AM Pipeline 1: End-to-end AM via LLM prompting, no step differentiation.
AM Pipeline 2: ADUR → ARE → map
AM Pipeline 3: ADUR → major ADU extraction → ARE → map containing major ADU and the relations between them

AM Pipeline 1:
Locked hyperparameters:
Generator: andrewelawrence/AMwithLLMs-Meta-Llama-3.1-8B-Instruct-bnb-4bit (base: unsloth/Meta-Llama-3.1-8B-bnb-4bit) per Argument mining for fine-tuned LLMs
Embedder: Qwen3-Embedding-0.6B per https://huggingface.co/spaces/mteb/leaderboard for instruction retrieval, speed, and size. Context Length: 32k, Prompting: instruction-aware retrieval
Vector store: FAISS IndexFlatIP per benchmarks https://unitech-selectedpapers.tugab.bg/images/2024/4-CST/s4_p72_v3.pdf) 
Prompt template (no prompt tuning as even simple prompt tuning requires SFT.)
Few-shot example (going with the one Aidan produces)
Similarity metric (cosine)
Reranker model (Qwen-3-reranker-0.6B)
Reasoning (CoT):
Prompt templates (just go with one for timing)
Reasoning depth (number steps)
Retrieval-in-reasoning (RA-CoT):
Retrieval step position (retrieve at ADU extraction step. Context then just passed along.)
Context fusion strategy (see above)
Decoding (Generation):
Temperature: do values near 0.7 per Can Large Language Models perform Relation-based Argument Mining? (January, 2025))
Max tokens: Do max near X per average token count of annotated argument map schemas
Search space hyperparameters:
Setup:
Zero-shot
Few-shot (n=1)
Zero-shot + RAG
Few-shot (n=1) + RAG
CoT
RA-CoT (RAG + CoT)
Embedder model dimension: 512/1024
Chunk size: 500-1000 tokens? Determine what it should be.
Chunk overlap: 100 - 150 tokens ish
RAG: Top-k retrieved + Score threshold. Reranker model top-n rerank

AM Pipeline 2:
Models: 
ADUR: chkla/roberta-argument, ArneBinder/sam-adur-sciarg
ARE: raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L, ArneBinder/sam-are-sciarg
Search space hyperparameters:
Model pairs:
ArneBinder/sam-adur-sciarg → ArneBinder/sam-are-sciarg
chkla/roberta-argument → raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L

AM Pipeline 3:
Models: 
ADUR: chkla/roberta-argument, ArneBinder/sam-adur-sciarg
ARE: raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L, ArneBinder/sam-are-sciarg
Search space hyperparameters:
Model pairs:
ArneBinder/sam-adur-sciarg → ArneBinder/sam-are-sciarg
chkla/roberta-argument → raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L
Major ADU Extraction Method (centroid vs. pairwise per Towards an Argument Mining Pipeline Transforming Texts to Argument Graphs) 

Execute a Bayesian search (via Optuna) over hyperparameters in each pipeline. 

Verification: After identifying each model’s best hyperparams, sample n=20/50/100 extractions from the train set and validate the output fidelity to the human annotated maps.

Notes: This is the most technically difficult and involved part of the process. We’re hoping to finish it by Friday, August 8th.



Phase 2 - fine-tuning model weighted vote for the best model

Goal: Determine which model is the best to use by learning a weighted consensus of the four pipeline outputs s.t. the cost function below is minimized

Timing: Try to finish a week after Phase 1, so by August 15th.

Data: 
Input: 1-5,000 randomly sampled papers without human annotations (80/20 train/val split).
Output: Span and edge labels.

Feature Engineering
Output: For each paper, compute a vector embedding of the extracted argument map over the spans and relation types.
Metadata: Vectorize category names and titles in the same embedding space (via concatenation).
Distance Metric: Cosine distance between the metadata vector and the argument map vector.

Cost function: Logistic regression where

di = distance from model i.

Loss: BCE of distance below some threshold (of cosine distance between vectors) we set to define a “good extraction”
Regularization: L2 on wi
Hyperparam Tuning: C, learning rate
Training Metric: auROC, Precision@k

Validation: The model with the highest weight  (also corresponding auROC/Precision@k) is selected as the model to use for extraction. From this top-ranked model, randomly sample n=20/50/100 extractions and assess precision/recall qualitatively.

