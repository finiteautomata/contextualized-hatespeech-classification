import pytest
from hatedetection.preprocessing import preprocess_tweet

def test_preprocessing_replaces_users():
    """
    Replaces handles with special token for user
    """
    text = "@perezjotaeme debería cambiar esto"

    assert preprocess_tweet(text) == "[USER] debería cambiar esto"

def test_preprocessing_replaces_users_twice():
    """
    Replaces handles with special token for user
    """
    text = "@perezjotaeme @perezjotaeme debería cambiar esto"

    assert preprocess_tweet(text) == "[USER] [USER] debería cambiar esto"

def test_preprocessing_replaces_urls():
    """
    Replaces urls with special token for url
    """
    text = "esto es muy bueno http://bit.ly/sarasa"

    assert preprocess_tweet(text) == "esto es muy bueno [URL]"

def test_shortens_repeated_characters():
    """
    Replaces urls with special token for url
    """
    text = "no entiendo naaaaaaaadaaaaaaaa"

    assert preprocess_tweet(text, shorten=2) == "no entiendo naadaa"

def test_shortens_laughters():
    """
    Replaces laughters
    """

    text = "jajajajaajjajaajajaja no lo puedo creer ajajaj"
    assert preprocess_tweet(text) == "jaja no lo puedo creer jaja"

def test_replaces_odd_quotation_marks():
    """

    Replaces “ -> "

    """
    text = "Pero pará un poco, “loquita”"

    assert preprocess_tweet(text) == 'Pero pará un poco, "loquita"'


def test_preprocessing_handles_hashtags():
    """
    Replaces hashtags with text
    """
    text = "esto es #UnaGenialidad"

    assert preprocess_tweet(text) == "esto es una genialidad"
