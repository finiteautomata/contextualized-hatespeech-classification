import torch
from hatedetection import BertForSequenceMultiClassification
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
            "finiteautomata/bert-contextualized-hate-speech-es",
            "finiteautomata/bert-contextualized-hate-category-es",
            use_context=True
        )



    @classmethod
    def load_noncontextualized_model(cls):
        """
        Convenient method to construct noncontextualized model
        """
        return cls(
            "finiteautomata/bert-non-contextualized-hate-speech-es",
            "finiteautomata/bert-non-contextualized-hate-category-es",
            use_context=False
        )


    def __init__(self, base_model_name, category_model_name, use_context=False):
        """
        Constructor for HateSpeechAnalyzer class
        """
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=2,
        )


        self.category_model = BertForSequenceMultiClassification.from_pretrained(
            category_model_name, num_labels=len(extended_hate_categories)
        )

        self.use_context = use_context
        max_length = 256 if use_context else 128

        self.tokenizer.model_max_length = max_length


    def predict(self, sentence, context=None):
        """

        """
        device = self.base_model.device

        args = []


        # If context, prepend it
        if context and self.use_context:
            args.append(context)
        args.append(preprocess_tweet(sentence))

        idx = self.tokenizer.encode(*args)
        # Reshape to be (1, L) and send to model d
        idx = torch.LongTensor(idx).view(1, -1).to(device)

        outs = self.base_model(idx)
        hateful = bool(outs.logits.argmax().item())

        if hateful:
            """
            Look for categories
            """
            category_output = self.category_model(idx)
            category_output = category_output.logits.detach().cpu().numpy()[0]

            # If logit of cat is > 0 this means sigmoid(logit) > 0.5.
            categories = [cat for cat, out in list(zip(extended_hate_categories, category_output > 0)) if out]

            calls_to_action = "CALLS" in categories
            if calls_to_action:
                categories.remove("CALLS")

            return HateClassificationOutput(hateful=hateful, calls_to_action=calls_to_action, categories=categories)

        return HateClassificationOutput(hateful=hateful)
