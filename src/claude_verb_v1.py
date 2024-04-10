import spacy
from spacy.matcher import Matcher
from ginza import *

def extract_verb_phrases(text: str) -> str:
    nlp = spacy.load('ja_ginza')
    doc = nlp(text)

    verb_phrases = []
    for token in doc:
        # 動詞だったら
        if token.pos_ == 'VERB':
            verb_phrase = []
            for child in token.children:
                print(child.text, child.dep_)
                # 目的語があれば動詞の前にappend
                if child.dep_ == 'obj' or child.dep_ == 'obl':
                    verb_phrase.append(child.text)
            verb_phrase.append(token.text)
            for child in token.children:
                # 動詞の後につづく補語的なものがあれば
                if child.dep_ == 'aux':
                    verb_phrase.append(child.text)
            verb_phrases.append(''.join(verb_phrase))
    return verb_phrases

def main():
    

    sentences = [
        "少年がサッカーをする",
        "難しい本を読む",
        "彼女は毎朝コーヒーを飲む",
        "学生が図書館で勉強をする",
        "昨日のテストで良い点をとった",
        "先週風邪をひいてしまった",
        "昨日怪我をした"
    ]

    for sentence in sentences:
        verb_phrases = extract_verb_phrases(sentence)
        print(f"Sentence: {sentence}")
        print("Extracted verb phrases:")
        for phrase in verb_phrases:
            print(phrase)
        print()

if __name__ == "__main__":
    main()
