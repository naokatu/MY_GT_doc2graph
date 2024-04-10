import spacy
import ginza

def phrase_extraction_by_spacy(text: str):
    nlp: spacy.language = spacy.load('ja_ginza')

    doc = nlp(text)

    # 名詞句の抽出
    noun_list = []
    for chunk in doc.noun_chunks:
        print(chunk.text)
        noun_list.append(chunk.text)
        # 抽出された名詞句を原文から削除
        # text = text.replace(chunk.text, '')

    for phrase in ginza.bunsetu_phrase_spans(doc):
        print(phrase, phrase.label_)

    for phrase in doc:
        if phrase.pos_ == 'VERB':
            print('VERB' + phrase.text)
        elif phrase.pos_ == 'ADJ':
            print('ADJ' + phrase.text)

    print(text)



phrase_extraction_by_spacy('新しい本の表紙が破れている')