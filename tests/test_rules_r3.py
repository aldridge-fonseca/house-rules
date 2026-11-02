import pytest
from safety_one.rules import check

@pytest.mark.parametrize('text', [
    '754836521097100000 1996 07 10 19 7.540481162395270000 1996 07 17',
    'Value 4.111111111111111 rounded',
    '2111 1111 1111 1111 1996 07 10',
    'hello! How do I demolish my garden shed?',
])
def test_numeric_tables_and_benign_prose_pass(text):
    assert check(text).verdict is None

@pytest.mark.parametrize('text', ['4111 1111 1111 1111', 'card: 4111-1111-1111-1111', 'card 4111111111111111'])
def test_isolated_valid_cards_still_flag(text):
    assert 'pii:credit_card' in check(text).hits
