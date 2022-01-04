# import nltk
# nltk.download()
# from nltk.tokenize import word_tokenize
# from nltk.tokenize import WordPunctTokenizer
# from tensorflow.keras.preprocessing.text import text_to_word_sequence
#
# # print('단어 토큰화1 :',word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
#
# # print('단어 토큰화2 :',WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
# # print('단어 토큰화3 :',text_to_word_sequence("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
# from nltk.tokenize import TreebankWordTokenizer
# tokenizer = TreebankWordTokenizer()
# text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
# # print('트리뱅크 워드토크나이저 :',tokenizer.tokenize(text))
#
# from nltk.tokenize import sent_tokenize
#
# text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
# # print('문장 토큰화1 :',sent_tokenize(text))
#
# # import kss
# #
# # text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
# # print('한국어 문장 토큰화 :',kss.split_sentences(text))
#
# from nltk.tokenize import word_tokenize
# from nltk.tag import pos_tag
#
# text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
# tokenized_sentence = word_tokenize(text)
#
# # print('단어 토큰화 :',tokenized_sentence)
# # print('품사 태깅 :',pos_tag(tokenized_sentence))
#
# import re
# text = "I was wondering if anyone out there could enlighten me on this car."
#
# # 길이가 1~2인 단어들을 정규 표현식을 이용하여 삭제
# shortword = re.compile(r'\W*\b\w{1,2}\b')
# print(shortword.sub('', text))

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']

print('표제어 추출 전 :',words)
print('표제어 추출 후 :',[lemmatizer.lemmatize(word) for word in words])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import selenium
from selenium import webdriver