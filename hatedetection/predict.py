import torch
from .preprocessing import preprocess_tweet
from .categories import extended_hate_categories

def predict_category(model, tokenizer, sentence, context=None):
    """
    Predicts sentence category


    Arguments:
    ----------

    model: BertForSequenceMultiClassification

    sentence: string
        Hateful tweet to be categorized

    context: string (Optional)


    """
    device = model.device

    args = []


    # If context, prepend it
    if context:
        args.append(context)
    args.append(preprocess_tweet(sentence))

    idx = tokenizer.encode(*args)
    # Reshape to be (1, L) and send to model device
    idx = torch.LongTensor(idx).view(1, -1).to(device)

    # Get logits
    output = model(idx)
    output = output.logits.detach().cpu().numpy()[0]

    # If logit of cat is > 0 this means sigmoid(logit) > 0.5.
    ret = [cat for cat, out in list(zip(extended_hate_categories, output > 0)) if out]

    return ret