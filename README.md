# NLP 101: 딥러닝과 자연어 처리 학습을 위한 자료 저장소
본 문서는 딥러닝을 이용한 자연어 처리를 학습하고자 하는 분들을 대상으로 작성되었습니다.
추가되었으면 좋겠다 생각하시는 자료를 알려주시면 반영하도록 하겠습니다.

_본 문서는 아래와 같은 규칙을 따라 작성되었습니다._
- 기본적으로 동일한 내용을 다루는 자료는 중복해서 기록하지 않습니다. <br/>
*e.g.) Dive into Deep Learning과 Deep Learning book 중 더 좋은 자료라고 판단한 Deep Learning book만 기록합니다.*
- 난이도가 유사하다고 판단되는 자료는 하나만 기록합니다.
- 다만, 유사 난이도를 보유한 자료가 한글 자료일 경우, 영어에 어려움이 있으신 분들을 고려해 함께 기록합니다.
- 난이도의 차이가 있는 자료, 이를테면 선후행 학습이 수반되어야 하는 자료는 모두 기록합니다.
<br/>

## Mathematics
### Statistics and Probabilities
| Source | Description |
|:---:|---|
| [Statistics 110](https://www.edwith.org/harvardprobability/) | 문과생도 이해할 수 있을 정도로 쉽게 확률론에 대한 설명을 해주는 강의입니다. |
| [확률과 통계](http://www.kocw.net/home/search/kemView.do?kemId=1056974) | KOCW에서 높은 평점을 자랑하는 한양대학교 이상화 교수님의 확률과 통계 강의입니다. |
<br/>

### Linear Algebra
| Source | Description |
|:---:|---|
| [Linear Algebra](https://www.youtube.com/watch?v=ZK3O402wf1c&list=PLE7DDD91010BC51F8) | Gilbert Strang 교수의 전설적인 선형대수 강의입니다. |
| [선형대수](http://www.kocw.net/home/search/kemView.do?kemId=977757) | KOCW에서 높은 평점을 자랑하는 한양대학교 이상화 교수님의 선형대수 강의입니다. |
| [인공지능을 위한 선형대수](https://m.edwith.org/linearalgebra4ai/lectures/14072) | 머신러닝에서 자주 사용되는 선형대수의 기초와 응용을 _(친절하게)_ 다루고 있는 스타 교수 주재걸 교수님의 강의입니다. |
| [Computational Linear Algebra](https://www.youtube.com/playlist?list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY) | fast.ai의 Rachel Thomas가 강의한 코드를 통해 이해하는 선형대수 강의입니다. 엔지니어 분들이 선형대수를 이해하는데 최적의 강의라고 생각합니다. |
| [Matrix methods in Data Analysis and Machine Learning](https://www.youtube.com/watch?v=Cx5Z-OslNWE&list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k) | Gilbert Strang 교수의 선형대수 응용편입니다. 선형대수를 선수 지식으로 하기에 난이도가 있지만, 실제 선형대수가 머신러닝에 어떻게 활용되는지 학습할 수 있는 좋은 강의입니다. |
<br/>

### Basic mathematics & Overview
| Source | Description |
|:---:|---|
| [Calculus](https://ocw.mit.edu/ans7870/resources/Strang/Edited/Calculus/Calculus.pdf) | Gilbert Strang 교수의 미적분학 교재입니다. 모든 챕터를 볼 필요는 없지만, Chapter 2-4, 11-13, 15-16 등은 학습하면 좋을 것 같다고 생각해 추가하였습니다. |
| [Mathematics for Machine Learning](https://mml-book.github.io/) | 머신러닝 학습에 수반되는 수학 지식을 모두 담은 책입니다. 개괄적 설명을 이어나가기에 이공계 학부 수준의 수학 지식은 선행되어야 이해하기 수월할 것이라 생각합니다. |
<br/>

## Deep Learning and Natural Language Processing
### Deep Learning
| Source | Description |
|:---:|---|
| [모두를 위한 딥러닝](https://www.youtube.com/watch?v=BS6O0zOGX4E&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm) | Clova AI를 리드하고 계신 김성훈님의 딥러닝 강의입니다. 입문 수준으로 최고의 강의입니다. |
| [모두를 위한 딥러닝2](https://www.youtube.com/channel/UCC76Jmsg6SAjdvphzGSJMBQ) | 앞서 언급한 김성훈님 강좌의 후속작입니다. Tensorflow와 PyTorch 버전이 각각 존재하며, 최신 코드로 설명을 진행하기 때문에 가치가 있다고 생각합니다. |
| [CS230](https://www.youtube.com/results?search_query=cs230) | 말이 필요없는, 최근 deeplearning.ai이라는 인공지능 교육 스타트업까지 설립한 Andrew Ng 교수님의 스탠포드 내 딥러닝 강의입니다. |
| [Deep Learning Book](https://www.deeplearningbook.org/) | GAN의 아버지, Ian Goodfellow 주도로 작성된 명서입니다. 원서를 읽는데 어려움이 없으시다면, 해당 책은 꼭 읽어보시길 추천합니다. |
<br/>

### Natural Language Processing 
| Source | Description |
|:---:|---|
| [한국어 임베딩](http://www.yes24.com/Product/Goods/78569687) | ratsgo라는 필명으로 유명한 이기창님의 자연어 처리 서적입니다. 제목은 한국어 '임베딩' 이지만 현대 자연어 처리의 근간이 되는 모든 지식을 함축하고 있는 좋은 책입니다. 특히 수식으로 가득하여 어려울 수 있는 내용들이 정말 간결한 설명으로 소개하고 있기 때문에 자연어 처리를 처음 접하는 분들에게는 좋은 입문 서적이 될수도, 자연어 처리를 접하기는 했지만 수식에 대한 정확한 이해가 부족했던 분들에게는 좋은 보충서가 될 수 있는 명저입니다. |
| [밑바닥부터 시작하는 딥러닝2](http://www.hanbit.co.kr/store/books/look.php?p_code=B8950212853) | 밑바닥 시리즈의 자연어 처리 버전입니다. 신경망 이론을 선수 지식으로 필요로 하기에 난이도가 살짝 있지만, 한국어로 번역된 혹은 한국어로 작성된 자연어 책 중 수준급의 책입니다. |
| [딥러닝을 이용한 자연어 처리 ](https://www.edwith.org/deepnlp) | GRU로 유명한 조경현 교수님이 D2 캠퍼스에서 강의하신 자연어 처리 강의입니다. 딥러닝 지식에 대한 복습 이후, 자연어 처리를 개괄적으로 설명해주기 때문에 딥러닝 기본 지식이 선수 지식으로 필요합니다. |
| [Neural Network Methods for NLP ](https://www.morganclaypool.com/doi/abs/10.2200/S00762ED1V01Y201703HLT037) | Yoav Goldberg가 작성한 딥러닝을 이용한 자연어 처리 전문 서적입니다. 위트있는 설명으로 핵심을 잘 짚어주는 명서입니다. |
| [Eisenstein's NLP Note](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf) |  머신러닝을 이용한 자연어 처리 뿐 아니라 자연어 처리를 학습하기 위해 필요한 기본적인 언어학 지식을 함께 다루는 명서입니다. |
| [CS224N ](https://www.youtube.com/watch?v=8rXD5-xhemo&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z) |  Stanford 대학의 자연어 처리 명강의입니다. 2019년 버전까지 나왔기 때문에 최신 트렌드까지 다룬다는 큰 장점이 있습니다. |
| [CS224U ](https://www.youtube.com/watch?v=tZ_Jrc_nRJY&list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20) |  올해 신설된 것으로 보이는 자연어 이해 강의입니다. CS224N 이후 수강하면 좋을 것 같아보이며, PyTorch로 과제를 제공한다는 점이 매력적입니다. |
| [Code-First Intro to Natural Language Processing](https://www.youtube.com/playlist?list=PLtmWHNX-gukKocXQOkQjuVxglSDYWsSh9) | fast.ai의 공동 설립자 Rachel Thomas가 진행하는 코드로 이해하는 자연어 처리 강의입니다. 강의를 듣다보면 Rachel Thomas가 내뿜는 Motivation에서 헤어나올 수 없게 됩니다. | 
| [Natural Language Processing with PyTorch](https://www.amazon.com/Natural-Language-Processing-PyTorch-Applications/dp/1491978236) | 양질의 데이터 과학 책을 출판하기로 유명한 O'REILLY 사의 자연어 처리 서적입니다. 기본 코드가 PyTorch로 작성되어 있으므로, PyTorch 유저분들이 읽기 좋은 책입니다. |
| [Linguistic Fundamentals for Natural Language Processing](https://www.amazon.com/Linguistic-Fundamentals-Natural-Language-Processing/dp/1627050116) | Bender rule로 유명한 언어학자 Emily Bender의 언어학 서적입니다. 딥러닝 관련 서적은 아니지만 언어학과 관련된 도메인 지식을 기를 수 있는 훌륭한 입문서입니다. |

<br/>

## Libraries related to the Natural Language Processing
| Source | Description |
|:---:|---|
| [NumPy](http://cs231n.github.io/python-numpy-tutorial/) | 머신러닝 연산에 필수적으로 사용되는 NumPy를 Stanford CS231N 강좌에서 정리해주었습니다. |
| [PyTorch](https://pytorch.org/tutorials/) | Facebook이 제공하는 PyTorch Tutorial로 양질의 퀄리티를 자랑합니다. |
| [Tensorflow](https://www.tensorflow.org/tutorials/text/word_embeddings) | Tensorflow에서 직접 제공하는 튜토리얼로, 최근 퀄리티가 급상승하였습니다. 기본적인 지식을 그림 자료와 함께 훌륭하고 설명합니다. |
| [spaCy](https://course.spacy.io/) | 최근 자연어 처리 분야에서 각광을 받고 있는 spaCy의 핵심 개발자 Ines가 작성한 튜토리얼입니다. |
| [torchtext](https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/) | PyTorch 사용 시, 손 쉽게 데이터 전처리가 가능한 torchtext의 튜토리얼입니다. 공식 문서보다 더 자세한 설명을 수반하고 있습니다. |
| [SentencePiece](https://github.com/google/sentencepiece) | Subword Information을 이용해 BPE 기반의 Vocabulary 구축을 도와주는 Google의 오픈 소스 라이브러리입니다. |
| [KoNLPy](http://konlpy.org/en/latest/) | 한국어 자연어 처리에 있어 중요하게 활용되는 여러 형태소 분석기를 포함하고 있는 라이브러리입니다. |
| [soynlp](https://github.com/lovit/soynlp) | 한국어 자연어 처리를 수행할 때 비지도 학습 기반의 여러 훈련을 가능케 해주는 라이브러리입니다. |
| [NLTK](https://datascienceschool.net/view-notebook/118731eec74b4ad3bdd2f89bab077e1b/) | 김도형 박사님이 제공하는 NLTK 튜토리얼로 보기도 편하며, 내용도 알찹니다. |
<br/>

## AWESOME blogs
| Blog | Article you should read |
|---|:---:|
| [Christopher Olah's Blog](https://colah.github.io/) | [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) |
| [Jay Alammar's Blog](http://jalammar.github.io/) | [Illustrated Word2vec](http://jalammar.github.io/illustrated-word2vec/) |
| [Chris McCormick's Blog](http://mccormickml.com/) | [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) |
| [Sebastian Ruder's Blog](http://ruder.io/) | [Tracking Progress in Natural Language Processing](https://nlpprogress.com/) |
| [The Gradient](https://thegradient.pub/) | [Evaluation Metrics for Language Modeling](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/) |
| [Thomas Wolf's Blog](https://medium.com/@Thomwolf) | [The Current Best of Universal Word Embeddings and Sentence Embeddings](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a) |
| [dair.ai](https://medium.com/dair-ai) | [A Light Introduction to Transfer Learning for NLP](https://medium.com/dair-ai/a-light-introduction-to-transfer-learning-for-nlp-3e2cb56b48c8) |
| [Machine Learning Mastery](https://machinelearningmastery.com/) | [How to Develop a Neural Machine Translation System from Scratch](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/) |
| [스캐터랩 핑퐁팀 블로그](https://blog.pingpong.us/) | [카톡 데이터는 어떻게 정제할 수 있을까?](https://blog.pingpong.us/dialog-bert-1/) |
| [김현중님 블로그](https://lovit.github.io/) | [Unsupervised tokenizers in soynlp project](https://lovit.github.io/nlp/2018/04/09/three_tokenizers_soynlp/) |
| [박상길님 블로그](http://docs.likejazz.com/) | [BERT 톺아보기](http://docs.likejazz.com/bert/) |
| [ratsgo님 블로그](https://ratsgo.github.io/) | [Word2Vec의 학습 방식](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/30/word2vec/) |
<br/>

## Communities
- [Allen AI NLP Highlights](https://allenai.org/podcasts/podcasts-all.html)
- [Tensorflow Korea](https://www.facebook.com/groups/TensorFlowKR/)
- [PyTorch Korea](https://www.facebook.com/groups/PyTorchKR/)
- [Keras Korea](https://www.facebook.com/groups/KerasKorea/)
- [Reinforcement Learning Korea](https://www.facebook.com/groups/ReinforcementLearningKR/)
- [AI Robotics Korea](https://www.facebook.com/groups/airoboticskr/)
- [모두의 연구소](http://home.modulabs.co.kr/)
- [바벨피쉬](https://www.facebook.com/groups/babelPish/)
- [챗봇 코리아](https://www.facebook.com/groups/ChatbotDevKR/)
- [월간 자연어 처리](https://www.facebook.com/monthly.nlp/)
- [GDG Seoul](https://www.facebook.com/groups/gdgseoul/)
- [GDG Pangyo](https://www.facebook.com/groups/gdgpangyo/)
- [Montreal.AI](https://www.facebook.com/MontrealAI/)
- [Artificial Intelligence & Deep Learning](https://www.facebook.com/groups/DeepNetGroup/)
<br/>

## NLP Specialists You should remember
*(not enumarted by rank)*

| Name | Description | Known for |
|---|---:|:---:|
| Kyunghyun Cho | Professor @NYU | [GRU](https://arxiv.org/abs/1406.1078) |
| Yejin Choi | Professor @Washington Univ. | [Grover](https://arxiv.org/abs/1905.12616) |
| Yoon Kim | Ph.D Candidate @Harvard Univ. | [CNN for NLP](https://www.aclweb.org/anthology/D14-1181) |
| Minjoon Seo | Researcher @Clova AI, Allen AI | [BiDAF](https://arxiv.org/abs/1611.01603) |
| Kyubyong Park | Researcher @Kakao Brain | [Paper implementation & NLP with Korean language](https://github.com/Kyubyong) |
| Tomas Mikolov | Researcher @FAIR | [Word2vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) |
| Omer Levy | Researcher @FAIR | [Various Word Embedding techniques](https://scholar.google.co.il/citations?user=PZVd2h8AAAAJ&hl=en) |
| Jason Weston | Researcher @FAIR | [Memory Networks](https://arxiv.org/abs/1410.3916) |
| Yinhan Liu | Researcher @FAIR | [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf) |
| Guillaume Lample | Researcher @FAIR | [XLM](https://arxiv.org/pdf/1901.07291.pdf) |
| Alexis Conneau | Researcher @FAIR | [XLM-R](https://arxiv.org/abs/1901.07291) |
| Ashish Vaswani | Researcher @Google | [Transformer](https://arxiv.org/abs/1706.03762) |
| Jacob Devlin | Researcher @Google | [BERT](https://arxiv.org/abs/1810.04805) |
| Matthew Peters | Researcher @Allen AI | [ELMo](https://arxiv.org/abs/1802.05365) |
| Alec Radford | Researcher @Open AI | [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| Sebastian Ruder | Researcher @DeepMind | [NLP Progress](https://nlpprogress.com/) |
| Richard Socher | Researcher @Salesforce | [Glove](https://www.aclweb.org/anthology/D14-1162) |
| Jeremy Howard | Co-founder @Fast.ai | [ULMFiT](https://arxiv.org/abs/1801.06146) |
| Thomas Wolf | Lead Engineer @Hugging face | [pytorch-transformers](https://github.com/huggingface/pytorch-transformers)
| Luke Zettlemoyer | Professor @Washington Univ. | [ELMo](https://arxiv.org/abs/1802.05365) |
| Yoav Goldberg | Professor @Bar Ilan Univ. | [Neural Net Methods for NLP](https://www.morganclaypool.com/doi/abs/10.2200/S00762ED1V01Y201703HLT037) |
| Chris Manning | Professor @Stanford Univ. | [CS224N](https://www.youtube.com/watch?v=8rXD5-xhemo&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z) |
| Dan Jurafsky | Professor @Stanford Univ. | [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)
| Graham Neubig | Professor @CMU | [Neural Nets for NLP](https://www.youtube.com/watch?v=pmcXgNTuHnk&list=PL8PYTP1V4I8Ajj7sY6sdtmjgkt7eo2VMs) |
| Zhilin Yang | Ph.D Candidate @CMU | [XLNet](https://arxiv.org/abs/1906.08237) |
| Abigail See | Ph.D Candidate @Stanford Univ. | [Pointer Generator](http://www.abigailsee.com/2017/04/16/taming-rnns-for-better-summarization.html) |
| Eric Wallace | Ph.D Candidate @Berkely Univ. | [AllenNLP Interpret](https://arxiv.org/abs/1909.09251) |

<br/>

## Research Conferences
- [ACL](https://www.aclweb.org/portal/)
- [NAACL](https://www.aclweb.org/anthology/venues/naacl/)
- [EMNLP](https://www.aclweb.org/anthology/venues/emnlp/)
- [ICML](https://icml.cc/)
- [ICLR](https://www.iclr.cc/)
- [NeurIPS](https://nips.cc/)
- [AAAI](http://www.aaai.org/)
- [EurNLP](https://www.eurnlp.org/)
- [한국어정보처리학회](http://www.kips.or.kr/)