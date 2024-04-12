import re
import urllib.request
import zipfile
import tarfile
import os
from typing import List, Dict, Any


def return_aozora_text(url: str, filename: str, start_row:int, end_row:int) -> str:
    # zipファイルのダウンロード
    urllib.request.urlretrieve(url,filename)
    with zipfile.ZipFile('./' + filename) as zip_file:
        zip_file.extractall('.')
    os.remove(filename)

    # テキストファイルから文字列を抽出
    with open('./meditationes.txt', 'r', encoding='shift_jis') as f:
        text = f.read()

    # 本文を抽出 (ヘッダとフッタを削除)
    text = ''.join(text.split('\n')[start_row:end_row]).replace('　', '')
    # 注釈 (［］) とルビ (《》) を削除
    text = re.sub(r'(［.*?］)|(《.*?》)', '', text)

    return text

# {'IT': ['1個目のファイルの文字列', '2個目のファイルの文字列', ], 'daily': ['ccc', 'ddd']}を返す
def return_livedoor_text(url: str, filename: str) -> Dict[str, List[Any]]:
    urllib.request.urlretrieve(url, filename)
    with tarfile.open('./' + filename, 'r:gz') as tar:
        tar.extractall('.')
    os.remove(filename)

    corpus_dir = 'text'
    text = ''
    text_li = {}
    for category in os.listdir(corpus_dir):
        target_dir = os.path.join(corpus_dir, category)
        # ファイルだったらスキップ
        if os.path.isfile(target_dir):
            continue
        text_li[target_dir] = []
        for file in os.listdir(target_dir):
            # 記事じゃないテキストファイルを除外
            if file in ['CHANGE.txt', 'LICENSE.txt', 'README.txt']:
                continue

            filename = os.path.join(target_dir, file)
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 最初の2行を削除
            content_lines = lines[2:]
            # 改行と空白削除
            content = re.sub(r'\s', '', ''.join(content_lines))

            text_li[target_dir].append(content)

        return text_li
