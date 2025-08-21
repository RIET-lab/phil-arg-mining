---
license: cc-by-nc-sa-4.0
language:
- en
tags:
- argument mining
datasets:
- US2016
- QT30
metrics:
- macro-f1
---
# Argument Relation Mining

Argument Mining model trained with English (EN) data for the Argument Relation Identification (ARI) task using the US2016 and the QT30 corpora. 

Extending the best performing RoBERTa-large model trained in the "Transformer-Based Models for Automatic Detection of Argument Relations: A Cross-Domain Evaluation" paper.

macro-F1 (4-class classification): **0.70**


Conf. Matrix on test:

| Class         | **None** | **Inference** | **Conflict** | **Rephrase** |
|---------------|----------|---------------|--------------|--------------|
| **None**      | 2991     | 133           | 13           | 24           |
| **Inference** | 139      | 547           | 51           | 103          |
| **Conflict**  | 38       | 54            | 98           | 21           |
| **Rephrase**  | 55       | 128           | 25           | 443          |    


Cite:

```
@article{ruiz2021transformer,
author = {R. Ruiz-Dolz and J. Alemany and S. Barbera and A. Garcia-Fornes},
journal = {IEEE Intelligent Systems},
title = {Transformer-Based Models for Automatic Identification of Argument Relations: A Cross-Domain Evaluation},
year = {2021},
volume = {36},
number = {06},
issn = {1941-1294},
pages = {62-70},
doi = {10.1109/MIS.2021.3073993},
publisher = {IEEE Computer Society}
}

```