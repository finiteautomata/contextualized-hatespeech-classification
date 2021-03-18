"""
These are the posible categories of hatespeech
"""
hate_categories = [
    "WOMEN", # Against women
    "LGBTI", # Against LGBTI
    "RACISM", # Racist
    "CLASS",  # Classist
    "POLITICS", # Because of politics
    "DISABLED", # Against disabled
    "APPEARANCE",  # Against people because their appearance
    "CRIMINAL", # Against criminals
]

"""
Categories + CALLS (call for action)
"""
extended_hate_categories = ["CALLS"] + hate_categories