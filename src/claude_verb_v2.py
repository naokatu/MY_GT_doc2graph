import spacy
from spacy.tokens import Token

def get_obj_phrases(token):
    phrases = []
    for child in token.children:
        if child.dep_ == 'obj' or child.dep_ == 'obl':
            phrase = ' '.join([w.text for w in child.subtree])
            phrases.append(phrase)
    return phrases

Token.set_extension('obj_phrases', getter=get_obj_phrases)

def extract_verb_phrases(doc):
    verb_phrases = []
    for token in doc:
        if token.pos_ == 'VERB':
            verb_phrase = []
            aux_tokens = []
            for child in token.children:
                if child.dep_ == 'aux':
                    aux_tokens.append(child.text)
            for obj_phrase in token._.obj_phrases:
                verb_phrase.append(obj_phrase)
            verb_phrase.append(token.text + ''.join(aux_tokens))
            verb_phrase = ' '.join(verb_phrase)
            if not verb_phrase.endswith('。'):
                verb_phrases.append(verb_phrase)
    return verb_phrases

def main():
    nlp = spacy.load('ja_ginza')
    sentences = [
        "少年がサッカーをする",
        "難しい本を急いで読む",
        "彼女は毎朝コーヒーを飲む",
        "学生が図書館で勉強をする",
        "昨日のテストで良い点をとった",
        "先週風邪をひいてしまった",
        "昨日怪我をした"
    ]
    for sentence in sentences:
        doc = nlp(sentence)
        verb_phrases = extract_verb_phrases(doc)
        print(f"Sentence: {sentence}")
        print("Extracted verb phrases:")
        for phrase in verb_phrases:
            print(phrase.replace(' ', ''))
        print()

if __name__ == "__main__":
    main()
