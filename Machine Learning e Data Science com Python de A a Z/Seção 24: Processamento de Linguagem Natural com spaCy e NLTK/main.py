import bs4 as bs
import urllib.request
import nltk
import spacy

"""
Marcação POS
- POS (part-of-speech) atribui para as palavras partes da fala, como substantivos, adjetivos, verbos
- Importante para a detecção de entidades no texto, pois primeiro é necessário saber o que o texto contém
- Lista de tokens: https://spacy.io/api/annotation#pos-tagging
- Português: https://www.sketchengine.eu/portuguese-freeling-part-of-speech-tagset/
"""

# TESTES
# pln = spacy.load('pt_core_news_sm')
# print(pln)
# print('---------------------------------------------')
# print('Marcação POS')
# print('FRASE: Estou aprendendo processamento de linguagem natural, curso em Curitiba')
# documento = pln('Estou aprendendo processamento de linguagem natural, curso em Curitiba')
# print(type(documento))
#
# for token in documento:
#     print(token.text, token.pos_)
#
#
# """
# Lematização e stemização
# """
# print('---------------------------------------------\n')
# print('Lematização')
# print('FRASE: Estou aprendendo processamento de linguagem natural, curso em Curitiba')
# for token in documento:
#     print(token.text, token.lemma_)
#
# print('---------------------------------------------\n')
# print('Stemização')
# nltk.download('rslp')
#
# stemmer = nltk.stem.RSLPStemmer()
# print(stemmer.stem('aprender'))
#
# for token in documento:
#     print(token.text, token.lemma_, stemmer.stem(token.text))


# Retirando dados da Wikipedia
dados = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial')
dados = dados.read()
dados_html = bs.BeautifulSoup(dados)
paragrafos = dados_html.find_all('p')

conteudo = ''

for p in paragrafos:
    conteudo += p.text.lower()

print(conteudo)
