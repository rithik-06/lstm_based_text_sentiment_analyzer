import re

def clean_text(text):
    """Lowercase, remove punctuation and non-alphabetic characters."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Example usage:
# cleaned = clean_text("This movie was AMAZING!!! 10/10")
# print(cleaned)  # Output: 'this movie was amazing' 