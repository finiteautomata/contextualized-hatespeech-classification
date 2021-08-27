import torch
from hatedetection.preprocessing import preprocess_tweet
from hatedetection.categories import extended_hate_categories
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import functional as F

class HateClassificationOutput:
    """
    Base class for classification output
    """
    def __init__(self, hateful, calls_to_action=None, categories=[]):
        """
        Constructor
        """
        self.hateful = hateful
        self.calls_to_action = calls_to_action
        self.categories = categories

    def __repr__(self):
        ret = f"{self.__class__.__name__}"

        if not self.hateful:
            ret += "(hateful=False)"
            return ret
        else:
            ret += f"(hateful=True, calls_to_action={self.calls_to_action}, categories={self.categories}"
            return ret

class HateSpeechAnalyzer:
    """
    Wrapper to use HS models as black-box
    """

    @classmethod
    def load_contextualized_model(cls):
        """
        Convenient method to construct contextualized model
        """
        return cls(
            "finiteautomata/beto-fine-grained-hatespeech-news",
            use_context=True
        )

    def __init__(self, base_model_name, use_context=False, device="cpu"):
        """
        Constructor for HateSpeechAnalyzer class
        """
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=len(extended_hate_categories)
        )

        self.base_model = self.base_model.to(device)
        self.use_context = use_context
        max_length = 256 if use_context else 128

        self.tokenizer.model_max_length = max_length


    def predict(self, sentence, context=None):
        """

        """
        device = self.base_model.device

        args = [preprocess_tweet(sentence)]
        # If context, prepend it
        if context and self.use_context:
            args.append(context)

        idx = self.tokenizer.encode(*args)
        # Reshape to be (1, L) and send to model d
        idx = torch.LongTensor(idx).view(1, -1).to(device)

        outs = self.base_model(idx)

        probas = torch.sigmoid(outs.logits[0]).detach().cpu().numpy()

        hateful = (probas[1:] > 0.5).sum()

        if hateful:
            """
            Look for categories
            """
            # If logit of cat is > 0 this means sigmoid(logit) > 0.5.
            categories = [cat for cat, out in list(zip(extended_hate_categories, probas > 0.5)) if out]

            calls_to_action = "CALLS" in categories
            if calls_to_action:
                categories.remove("CALLS")

            return HateClassificationOutput(hateful=hateful, calls_to_action=calls_to_action, categories=categories)

        return HateClassificationOutput(hateful=hateful)
