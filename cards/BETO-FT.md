---
language:
  - es
metrics:
  - f1
pipeline_tag: text-classification
---

## BETO

Model trained to detect hate speech comments in news articles. Base model is BETO, a Spanish BERT pre-trained model. The task the model was trained on is a multilabel classification problem, where each input have a label for each of the considered groups:

| Label      | Description                             |
| :--------- | :-------------------------------------- |
| WOMEN      | Against women                           |
| LGBTI      | Against LGBTI                           |
| RACISM     | Racist                                  |
| CLASS      | Classist                                |
| POLITICS   | Because of politics                     |
| DISABLED   | Against disabled                        |
| APPEARANCE | Against people because their appearance |
| CRIMINAL   | Against criminals                       |

There is an extra label `CALLS`, which represents whether a comment is a call to violent action or not.

## Input

The model was trained taking into account both the comment and the context. To feed this model, use the template

```python
TEXT [SEP] CONTEXT
```

where `[SEP]` is the special token used to separate the comment from the context.

### Example

If we want to analyze

```
Comment: Hay que matarlos a todos!!! Nos infectaron con su virus!
Context: China prohibió la venta de perros y gatos para consumo humano
```

The input should be

```python
Hay que matarlos a todos!!! Nos infectaron con su virus! [SEP] China prohibió la venta de perros y gatos para consumo humano
```

## Usage:

Sadly, the `huggingface` pipeline does not support multi-label classification, so this model cannot be tested directly in the side widget.

To use it, you can try our [Demo](https://huggingface.co/spaces/piubamas/discurso-de-odio).

If you want to use it with your own code, you can use the following snippet:

```python

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "piubamas/beto-contextualized-hate-speech"
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

id2label = [model.config.id2label[k] for k in range(len(model.config.id2label))]

def predict(*args):
    encoding = tokenizer.encode_plus(*args)

    inputs = {
        k:torch.LongTensor(encoding[k]).reshape(1, -1) for k in {"input_ids", "attention_mask", "token_type_ids"}
    }

    output = model.forward(
        **inputs
    )

    chars = list(zip(id2label, list(output.logits[0].detach().cpu().numpy() > 0)))

    return [char for char, pred in chars if pred]

context = "China prohíbe la cría de perros para consumo humano")
text = "Chinos hdrmp hay que matarlos a todos")

prediction = predict(text, context)
```

## Citation

```bibtex
@misc{perez2022contextual,
  doi = {10.48550/ARXIV.2210.00465},

  url = {https://arxiv.org/abs/2210.00465},
  author = {Pérez, Juan Manuel and Luque, Franco and Zayat, Demian and Kondratzky, Martín and Moro, Agustín and Serrati, Pablo and Zajac, Joaquín and Miguel, Paula and Debandi, Natalia and Gravano, Agustín and Cotik, Viviana},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Assessing the impact of contextual information in hate speech detection},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
