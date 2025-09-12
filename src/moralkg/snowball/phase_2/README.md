Purpose
- Phase 2: learn weights over pipeline outputs using maps from unlabeled data and metadata similarity; select best model.

Intended workflow
- Build vector embeddings of predicted maps (spans + relation types) and metadata (category/title) in shared space.
- Compute cosine distances between metadata vector and each model’s map vector.
- Learn logistic regression (labels: distance < threshold) with L2 regularization (could also consider L1 as it tends to go to 0 for weights - which would be good for us - eliminating methods); tune C and learning rate.
- Select model with highest learned weight or just sample maps for final dataset based on weighting; validate on samples.

Config usage
- snowball.phase_2.distance_thr, sample_size, rand_sample_size, holdout
- snowball.phase_2.hparams.C (grid)

Status
- Evaluation and training utilities under phase_2/evals are placeholders; core training loop not implemented.
