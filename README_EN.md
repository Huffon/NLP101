# NLP 101: a Resouce Repository for Deep Learning and Natural Language Processing
This document is drafted for those who have enthusiasm for Deep Learning in natural language processing. If there are any good recommendations or suggestions, I will try to add more.

_This document is drafted with the rules as follows:_
- Materials that are considered to cover the same grounds will not be recorded repeatedly.
- Only one among those within similar level of difficulty will be recorded.
- Materials with different level of difficulty that need prerequsite or additional learning will be recorded.

Language: [Korean](/README.md) | [English](/README_EN.md)

<br/>

## Mathematics
#### Statistics and Probabilities
| Source | Description |
|:---:|---|
| [Statistics 110](https://www.edwith.org/harvardprobability/) | A lecture on Probability that can be easily understood by non-engineering major students. |
| [Brandon Foltz's Statistics](https://www.youtube.com/user/BCFoltz/playlists) | Brandon Foltz's Probability and Statistics lectures are posted on Youtube and is rather short, so it can be easily accessed during daily commute. |

<br/>

### Linear Algebra
| Source | Description |
|:---:|---|
| [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | A Linear algebraic lecture on Youtube channel 3Blue1Brown. Could be a big help for those planning to take undergraduate-level linear algebra since it allows overall understanding. It provides intutitively understandable visual aids to getting the picture of Linear algebra. |
| [Linear Algebra](https://www.youtube.com/watch?v=ZK3O402wf1c&list=PLE7DDD91010BC51F8) | A legendary lecture of professor Gilbert Strang. |
| [Matrix methods in Data Analysis and Machine Learning](https://www.youtube.com/watch?v=Cx5Z-OslNWE&list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k) | Professor Gilbert Strang's lecture on applied Linear algebra. As Linear algbra is prerequisite knowledge here, it is quite difficult to understand yet a great lecture to learn how Linear algebra is actually applied in the field of Machine Learning. |

<br/>

### Basic mathematics & Overview
| Source | Description |
|:---:|---|
| [Essence of calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) | A calculus lecture by the channel 3Blue1Brown mentioned above, helpful for those who want an overview of calculus likewise. | 
| [Calculus](https://ocw.mit.edu/ans7870/resources/Strang/Edited/Calculus/Calculus.pdf) | A coursebook on calculus written by professor Gilbert Strang. There is no need to go through the whole book, but chapters 2-4, 11-13, 15-16 are very worth studying. |
| [Mathematics for Machine Learning](https://mml-book.github.io/) | A book on all the mathematical knowledge accompanied with machine learning. Mathematic knowledge within the collegiate level of natural sciences or engineering is preferable here, as the explanations are mainly broad-brush. |

<br/>

## Deep Learning and Natural Language Processing
### Deep Learning
| Source | Description |
|:---:|---|
| [CS230](https://www.youtube.com/results?search_query=cs230) | A Deep Learning lecture of the renouned professor Andrew Ng, who has recently founded a startup on AI education. |
| [Deep Learning Book](https://www.deeplearningbook.org/) | A book written by Ian Goodfellow, the father of GAN, and other renouned professors. |
| [Dive into Deep Learning](https://d2l.ai/) | While the 'Deep Learning Book' above has theoretical explanation, this book also includes the codes to check how the notion is actually immplemented. |
| [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning) | Teaches readers how to write basic elements of the neural network with NumPy, without using Deep Learning Frameworks. Also a good material to study how high-level APIs work under the hood. |

<br/>

### Natural Language Processing 
| Source | Description |
|:---:|---|
| [Neural Network Methods for NLP ](https://www.morganclaypool.com/doi/abs/10.2200/S00762ED1V01Y201703HLT037) | An NLP book using Deep Learning written by Yoav Goldberg. It has witty explanations that lead to the fundamentals. |
| [Eisenstein's NLP Note](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf) | Awesome book to read that deals with not only NLP with machine learning, but also the basic linguistic knowledge to understand it. Eisenstein's book [Introduction to Natural Language Processing](https://www.amazon.com/Introduction-Language-Processing-Adaptive-Computation/dp/0262042843) was published based on this note. |
| [CS224N ](https://www.youtube.com/watch?v=8rXD5-xhemo&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z) | Awesome NLP lecture from Stanford. It has the 2019 version, dealing with the latest trends.|
| [CS224U ](https://www.youtube.com/watch?v=tZ_Jrc_nRJY&list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20) | An NLP lecture that was revalued since the advent of GLUE benchmark. Recommended to be taken after CS224N, and its merit is that it provides exercises in Pytorch. |
| [Code-First Intro to Natural Language Processing](https://www.youtube.com/playlist?list=PLtmWHNX-gukKocXQOkQjuVxglSDYWsSh9) | A code-first NLP lecture by Rachel Thomas, the co-founder of fast.ai. The motivation that Rachel Thomas gives is mind blowing. | 
| [Natural Language Processing with PyTorch](https://www.amazon.com/Natural-Language-Processing-PyTorch-Applications/dp/1491978236) | An NLP book from O'REILLY, known for numerous data science books of great quality. It is PyTorch-friendly as all the codes are written in PyTorch. |
| [Linguistic Fundamentals for Natural Language Processing](https://www.amazon.com/Linguistic-Fundamentals-Natural-Language-Processing/dp/1627050116) | A Linguistics book written by the linguist Emily Bender, known for Bender rule. Although not Deep Learning related, it is a great beginner's book on linguistic domain knowledge. |
<br/>

## Libraries related to the Natural Language Processing
| Source | Description |
|:---:|---|
| [NumPy](http://cs231n.github.io/python-numpy-tutorial/) | Stanford's lecture CS231N deals with NumPy, which is fundamental in machine learning calculations. |
| [Tensorflow](https://www.tensorflow.org/tutorials/text/word_embeddings) | A tutorial provided by Tensorflow. It gives great explanations on the basics with visual aids. |
| [PyTorch](https://pytorch.org/tutorials/) | An awesome tutorial on Pytorch provided by Facebook with great quality. |
| [tensor2tensor](https://github.com/tensorflow/tensor2tensor) | Sequence to Sequence tool kit by Google written in Tensorflow. |
| [fairseq](https://github.com/pytorch/fairseq) | Sequence to Sequence tool kit by Facebook written in Pytorch. |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | A library based on Transformer provided by Hugging Face that allows easy access to pre-trained models. One of the key NLP libraries to not only developers but researchers as well. |
| [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) | A tokenizer library that Hugging Face maintains. It boosts fast operations as the key functions are written in Rust. The latest tokenizers such as BPE can be tried out with Hugging Face tokenizers. |
| [spaCy](https://course.spacy.io/) | A tutorial written by Ines, the core developer of the noteworthy spaCy. |
| [torchtext](https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/) | A tutorial on torchtext, a package that makes data preprocessing handy. Has more details than the official documentation. |
| [SentencePiece](https://github.com/google/sentencepiece) | Google's open source library that builds BPE-based vocabulary using subword information. |

<br/>

## AWESOME blogs
| Blog | Article you should read |
|---|:---:|
| [Christopher Olah's Blog](https://colah.github.io/) | [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) |
| [Jay Alammar's Blog](http://jalammar.github.io/) | [Illustrated Word2vec](http://jalammar.github.io/illustrated-word2vec/) |
| [Sebastian Ruder's Blog](http://ruder.io/) | [Tracking Progress in Natural Language Processing](https://nlpprogress.com/) |
| [Chris McCormick's Blog](http://mccormickml.com/) | [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) |
| [The Gradient](https://thegradient.pub/) | [Evaluation Metrics for Language Modeling](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/) |
| [Distill.pub](https://distill.pub/) | [Visualizing memorization in RNNs](https://distill.pub/2019/memorization-in-rnns/) |
| [Thomas Wolf's Blog](https://medium.com/@Thomwolf) | [The Current Best of Universal Word Embeddings and Sentence Embeddings](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a) |
| [dair.ai](https://medium.com/dair-ai) | [A Light Introduction to Transfer Learning for NLP](https://medium.com/dair-ai/a-light-introduction-to-transfer-learning-for-nlp-3e2cb56b48c8) |
| [Machine Learning Mastery](https://machinelearningmastery.com/) | [How to Develop a Neural Machine Translation System from Scratch](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/) |

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
| Nikita Kitaev | Ph.D Candidate @UC Berkeley | [Reformer](https://arxiv.org/abs/2001.04451) | 
| Zihang Dai | Ph.D Candidate @CMU | [Transformer-XL](https://arxiv.org/abs/1901.02860) |
| Zhilin Yang | Ph.D Candidate @CMU | [XLNet](https://arxiv.org/abs/1906.08237) |
| Abigail See | Ph.D Candidate @Stanford Univ. | [Pointer Generator](http://www.abigailsee.com/2017/04/16/taming-rnns-for-better-summarization.html) |
| Eric Wallace | Ph.D Candidate @Berkely Univ. | [AllenNLP Interpret](https://arxiv.org/abs/1909.09251) |

<br/>

## Research Conferences
- [ACL](https://www.aclweb.org/portal/)
- [AAAI](http://www.aaai.org/)
- [CoNLL](https://www.conll.org/)
- [EMNLP](https://www.aclweb.org/anthology/venues/emnlp/)
- [EurNLP](https://www.eurnlp.org/)
- [ICLR](https://www.iclr.cc/)
- [ICML](https://icml.cc/)
- [IJCAI](https://www.ijcai.org/)
- [NAACL](https://www.aclweb.org/anthology/venues/naacl/)
- [NeurIPS](https://nips.cc/)
