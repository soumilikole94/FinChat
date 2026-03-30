from finchat.ui import _escape_currency_markdown


def test_escape_currency_markdown_escapes_dollar_signs():
    text = "Revenue rose to $35 billion and EPS reached $0.81."

    escaped = _escape_currency_markdown(text)

    assert escaped == r"Revenue rose to \$35 billion and EPS reached \$0.81."
