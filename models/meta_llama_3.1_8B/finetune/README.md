---
library_name: peft
license: other
base_model: unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: AMwithLLMs-Meta-Llama-3.1-8B-Instruct-bnb-4bit
  results: []
---

# AMwithLLMs-Meta-Llama-3.1-8B-Instruct-bnb-4bit

This model is a fine-tuned version of [unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit) on the Persuasive Essays (PE), Cornell eRulemaking Corpus (CDCP), and Abstracts of Randomized Control Trials (AbstRCT) datasets. It implements the fine-tuning process as described in [Argument Mining with Fine-Tuned Large Language Models](https://aclanthology.org/2025.coling-main.442/) (Cabessa et al., COLING 2025) and availabile at [https://github.com/mohammadoumar/AMwithLLMs](https://github.com/mohammadoumar/AMwithLLMs).

### Citation

```
@inproceedings{cabessa-etal-2025-argument,
    author = "Cabessa, Jeremie and Hernault, Hugo and Mushtaq, Umer",
    title = "Argument Mining with Fine-Tuned Large Language Models",
    publisher = "Association for Computational Linguistics",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    editor = "Rambow, Owen and Wanner, Leo and Apidianaki, Marianna and Al-Khalifa, Hend and Eugenio, Barbara Di and Schockaert, Steven",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    url = "https://aclanthology.org/2025.coling-main.442/",
    pages = "6624--6635",
}
```

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- total_eval_batch_size: 32
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 5
- mixed_precision_training: Native AMP

### Framework versions

- PEFT 0.15.2
- Transformers 4.52.4
- Pytorch 2.7.1+cu126
- Datasets 3.6.0
- Tokenizers 0.21.1
