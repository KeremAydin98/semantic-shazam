# semantic-shazam

The dataset:
https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres

My version of dataset and models:
https://drive.google.com/drive/folders/1hZw8nHlHS2ww_Hywksj2G3d1_5kotPD6?usp=sharing

## Introduction

This project aims to classify and summarize songs based on their lyrics. That's why I called this project Semantic Shazam, since it builds on the meaning of the song lyrics instead of the music sound. For the given song lyric, it returns the summarization of the song lyrics, most similar songs in the dataset and finally genre of the song.

<p align="center">
  <img src="https://user-images.githubusercontent.com/77073029/203484003-82b8d37f-f140-4f63-ac78-d356b9d63549.jpg" />
</p>

## Extracting semantics

But how do we make the computer understand meaning? Word embeddings are the solution of the researchers. They capture the meaning by comparing neighboring words, since similar words would have the same context words. That is actually what humans do in case we do not know the meaning of a word in a sentence. We look at other words of that sentence and try to derive a meaning. 

Word2Vec is the algorithm that I have just explained. It is used to train word embeddings. Word embeddings are just fixed sized vectors. Train set is extracted with a sliding windows that goes from left to right. The algorithm tries to predict the center word' vector from the neighboring words' vectors(it can also be done in the opposite way). This is called Continuous Bag of Words method. 


![1 YvOdGp73pOHmYGHKqkx5wQ](https://user-images.githubusercontent.com/77073029/203485283-e6624ad6-8f00-45e0-9a5d-3506bc54be6d.png)


As Jay Alammar stated, these vectors can derive interesting connections between words such as "king - man + woman = queen" in his great explanation of Word2Vec algorithm:

https://jalammar.github.io/illustrated-word2vec/

Just like that we derived the meaning of words, but I needed to find connections between songs. Therefore, I also needed paragraph vectors for songs besides word vectors. For these kind of tasks, Doc2Vec algorithm is utilized. The method is very similar to the Word2Vec. There is only one difference which is that with every prediction of the center word another vector called paragraph vector is trained next to the neighboring word vectors.

<p align="center">
  <img src="https://user-images.githubusercontent.com/77073029/203486637-433330ad-2176-442f-a948-f684e79c6ee0.png" />
</p>

**Gensim** library already had a doc2vec implementation. I only had one adjustment to the train set. Not every word add a meaning to the sentence like "the", "a" or etc. These kind of words are called stop words in the literature. I have removed them by using **spacy** library. 

## What can Semantic Shazam do?

### 1. Genre classification

The dataset contained many unique genres. So I have only chosen the most frequent ones. 
The main genre list contained:

"Pop, Romântico, Pop/Rock, R&B, Rap, Electronica, Rock, Blues, Jazz, Folk, Country, Heavy Metal". 

The inputs for the model were the paragraph vectors of the songs that we derived while training with Doc2Vec algorithm. And the desired output is the genre of the song. I have turned both of the inputs and outputs to the tf.data.Dataset format and did a batching process on them. The structure of ANN model for the classification task can be seen below:

```
model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(output_size, activation="softmax"),
    ])

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
```

The loss is Categorical Cross Entropy because I turned genres into one hot encoding format using tf.one_hot method. 

The model reached over %60 accuracy, it does not seem a lot but you have to remember that the model is only guessing song genres from the words in the lyrics. Also some of the genres are very close to each other, it is sometimes hard to distinguish them. The confusion matrix of the model's predictions can be seen below:

![genre_classifier_cm](https://user-images.githubusercontent.com/77073029/203489995-ff18a3de-54da-43ff-b9ad-1b5672191739.png)

### 2. Most similar songs

The Doc2Vec class of Gensim already had a most_similar method that brings the closest paragraph vectors of the given song.

### 3. Summarization 

This was the hardest part of this project. Summarization is not an easy task in NLP. Especially when you do not have the needed dataset. The song dataset did not contain a summarization column, so I have added one. By using a simple method that utilizes the strength of the sentence according to the frequency of the words in it excluding the stop words of course. 

```
def add_summarization(doc):

  doc = doc.replace("\n",". ")
  doc = doc.replace("..",".")
  doc = doc.replace(",.",".")

  replace_abbreviations = {"'m":" am", "'ll":" will", "'em":" them",
                           "'re":" are", "'s":" is", "n't":" not" ,
                           "’m":" am", "’ll":" will", "’em":" them",
                           "’re":" are", "’s":" is", "n’t":" not" , }

  for k,v in replace_abbreviations.items():

    doc = doc.replace(k,v)

  doc = nlp(doc)

  keyword = []
  stop_words = list(STOP_WORDS)
  pos_tag = ["PROPN", "ADJ", "NOUN", "VERB"]

  for token in doc:

    if(token.text in stop_words or token.text in string.punctuation):
      continue

    if(token.pos_ in pos_tag):
      keyword.append(token.text)

  freq_word = Counter(keyword)

  try:

    max_freq = Counter(keyword).most_common(1)[0][1]
    for word in freq_word.keys():
      freq_word[word] = (freq_word[word]/max_freq)

    sent_strength = {}
    for sent in doc.sents:
      for word in sent:
        if word.text in freq_word.keys():
          if sent in sent_strength.keys():
            sent_strength[sent] += freq_word[word.text]
          else:
            sent_strength[sent] = freq_word[word.text]

    summarized_sentences = nlargest(1, sent_strength, key=sent_strength.get)

    final_sentences = [w.text for w in summarized_sentences]

    return " ".join(final_sentences)

  except:

    return None
```

This sometimes gave dubious summarizations but I needed to start from somewhere. So I have created a dataset for the summarization, what is next?

Now it is time to build the model. Summarization requires a Sequence to sequence model, since it takes a sequence as an input and returns a sequence as an output. Sequence to sequences model is built by using two different NLP models which are encoders and decoders. Encoders are similar the sequential labelling models and decoder are similar to language models. 

For a summarization first we need to derive the context from the input, then generate a summary from that.

Sequential labelling models takes a text as input and normally tags the words in that sentence with an RNN structure. But in encoders tagging of the words is not needed, instead the context is needed. We know that RNN structures store some kind of context from their previous inputs in their hidden state. So the hidden state of the last RNN cell in the encoder would contain context from every single word in that sentence which is what we need.

Language models are the ones that take this context vectors and generate some kind of summary from that context. And every single output of the language model is used to derive a cross entropy loss. The model compares the output with the excepted summary in our dataset. And backpropagates that loss to improve the performance of both encoder and decoder. 

![adada](https://user-images.githubusercontent.com/77073029/203494954-6607fe17-c7b2-4fa8-9039-98aa0d2522ca.png)

I know that the image above is a machine translation task with a seq2seq model. But since both of them has the same structure I think it is ok. There is one problem left. In training, we feed the language model the correct previous word from the labels in the dataset, however in prediction stage we do not have that. That's why we have to build a different model for inference. In prediction stage, we feed the language model its previous outputs as input instead of the correct input.

## An example result

Random song name: Kitty Box
Artist name:  Lil' Kim
Given genre:  Rap 

1/1 [==============================] - 0s 30ms/step

Predictions:
1.  Rap
2.  R&B
3.  Pop 

Given summary:
(My kitty box, my kitty box, my kitty box, my kitty box). .
1/1 [==============================] - 1s 1s/step
1/1 [==============================] - 0s 42ms/step
1/1 [==============================] - 0s 38ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 29ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 30ms/step
Predicted summary:
 i am gonna be a hesitate line

Most similar songs:
1.  Loverboy (Remix) (Feat. da Brat, Ludacris, Shawnna, & Twenty Ii)
2.  Freaky Deaky (With Doja Cat)
3.  Make It Nasty
--------------------------------------------
