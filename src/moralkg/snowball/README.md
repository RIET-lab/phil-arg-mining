Purpose
- Experiments to select and weight AM pipelines over two phases (see phase_1 and phase_2).

Structure
- phase_1: evaluate candidate pipelines against curated annotations; search hyperparams using composite loss.
- phase_2: learn a weighted consensus over pipeline outputs on unlabeled papers; select best model.

Config usage
- snowball.phase_1.*: eval thresholds, loss weights, decoding/cot/rag hparams.
- snowball.phase_2.*: distance threshold, sample sizes, regularization search space.
