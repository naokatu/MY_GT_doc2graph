import spacy
from typing import List

def noun_phrases_extraction_by_spacy(text: str) -> List[str]:
    nlp: spacy.language = spacy.load('ja_ginza')

    doc = nlp(text)

    # 名詞句の抽出　あまりに短い文章だと精度が落ちるかも
    noun_list = []
    for chunk in doc.noun_chunks:
        print(chunk.text)
        noun_list.append(chunk.text)
        # 抽出された名詞句を原文から削除
        # text = text.replace(chunk.text, '')

    print(noun_list)
    return noun_list, text

def verb_phrases_extract_by_spacy(text: str) -> List[str]:
    nlp = spacy.load('ja_ginza')
    doc = nlp(text)

    verb_phrases = []
    for token in doc:
        # 動詞だったら
        if token.pos_ == 'VERB':
            verb_phrase = []

            for child in token.children:
                # 目的語があれば動詞の前にappend
                if child.dep_ == 'obj' or child.dep_ == 'obl':
                    verb_phrase.append(child.text)
            verb_phrase.append(token.text)

            for child in token.children:
                # 動詞の後につづく補語的なものがあれば
                if child.dep_ == 'aux':
                    verb_phrase.append(child.text)

            verb_phrases.append(''.join(verb_phrase))
            # 抽出された動詞句を原文から削除
            # text = text.replace(verb_phrase, '')
            
    print(verb_phrases)
    return verb_phrases, text

def adj_phrases_extraction_by_spacy(text: str) -> List[str]:
    nlp = spacy.load('ja_ginza')
    doc = nlp(text)

    adj_phrases = []
    for token in doc:
        # 前後のtokenを取得
        next_token = token.doc[token.i+1] if not token.is_sent_end else None
        prev_token = token.doc[token.i-1] if not token.is_sent_start else None
        if next_token is not None:
            # 品詞を抽出
            token_tag = token.tag_.split('-')[0]
            if len(token.tag_.split('-')) >= 2 and len(token.tag_.split('-')) >= 2: 
                token_tag_sub = token.tag_.split('-')[1]
                next_token_tag_sub = next_token.tag_.split('-')[1]
            # 形容動詞 （形状詞 + な） ... 静かな
            if (token_tag == '形状詞') and (next_token.text == 'な'):
                adj_phrases.append(token.text + 'な')
            # 名詞 + な  ... 元気な
            if (token_tag == '名詞') and (next_token.text == 'な'):
                adj_phrases.append(token.text + 'な')
            # 動詞 + 形容詞的 ... わかりやすい, 怒りっぽい
            if (token_tag == '動詞') and (next_token_tag_sub == '形容詞的'):
                adj_phrases.append(token.text + next_token.text)
            # 単体の形容詞
            if (token_tag == '形容詞'):
                adj_phrases.append(token.text)

        if prev_token is not None:
            prev_token_tag = prev_token.tag_.split('-')[0]
            # 名詞 + ない ... 問題ない, 仕方ない
            if (prev_token_tag == '名詞') and (token_tag == '形容詞') and (token_tag_sub == '非自立可能'):
                adj_phrases.append(prev_token.text + token.text)
    
    print(adj_phrases)
    return adj_phrases


input_text = '彼女は毎朝熱いコーヒーを飲む'
noun_phrases_extraction_by_spacy(input_text)
adj_phrases_extraction_by_spacy(input_text)
verb_phrases_extract_by_spacy(input_text)

