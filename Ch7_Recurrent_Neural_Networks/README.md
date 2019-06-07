## Exercise 1: Many2One: Text Classification

- A Fast Text Solution for text(sentence) classification is [here](https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_classification). 
- Please understand how it work and use LSTM to re-implement it.
- Please read the toolboxes for text processing and iterating, you will need them someday: [tl.prepro](https://tensorlayer.readthedocs.io/en/latest/modules/prepro.html#sequence), [tl.iterate](https://tensorlayer.readthedocs.io/en/latest/modules/iterate.html), [tl.nlp](https://tensorlayer.readthedocs.io/en/latest/modules/nlp.html)

## Exercise 2: Many2Many: Language Modeling or Text Generation 

- 1) Language Modeling for PTB dataset
    - Understand [tl.iterate.ptb_iterator](https://tensorlayer.readthedocs.io/en/latest/modules/iterate.html#tensorlayer.iterate.ptb_iterator) and what will happen if batch size is too large or too small?
    - Load PTB dataset
```python
train_data, valid_data, test_data, vocab_size = tl.files.load_ptb_dataset()
```
   - Solution is [here](https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_ptb/tutorial_ptb_lstm.py)

- 2) Text Generation using “the Trump dataset” 
    - Solution is [here](https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_generation/tutorial_generate_text.py)

## Exercise 3: NLP Project (Optional)
- You can propose an NLP application or choice one on below:
- [Seq2Seq: Chatbot](https://github.com/tensorlayer/seq2seq-chatbot)
- [Seq2Seq: Machine Translation](https://github.com/tensorlayer/seq2seq-chatbot)
- [One2Many: Image Captioning](https://github.com/zsdonghao/Image-Captioning)
