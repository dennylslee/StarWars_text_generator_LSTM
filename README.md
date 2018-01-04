# Introduction

This project uses LSTM as a simple character generator. The character model (RNN LSTM based) is first built from learning a compiled text of Star Wars plots as posted on wikipedia. 

Three main programs are contained in this project:
1. char-based-data-prep.py for preparing the input data. 
2. char-based-text-gen-LSTM.py for constructing the LSTM model.
3. char-based-run-model.py for using the built model to generate text based on a lead-off phrase.

The LSTM is built on Keras framework. For some of the best LSTM tutorials I have encontered, go to [Colah's Blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) and [Shi Yan blog on understanding LSTM](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714)

## Data Preparation

The raw text file (Star Wars plots from wikipedia) is read in as a time series of characters and construct a sequence of character vector.  The variable "length" controls the vector size (i.e. timesteps) for the RNN-LSTM to process. 

The main code construct:

```python
# organize into sequences of characters
length = 10
sequences = list()
for i in range(length, len(raw_text)):
	# select sequence of tokens
	seq = raw_text[i-length:i+1]
	# store
	sequences.append(seq)
print('Total Sequences: %d' % len(sequences))
```


## LSTM model building

A single layer LSTM is used for performing the character prediction.  The LSTM is based on a many-to-one architecture.

The internal state (vector) size of the cell and hidden state is set as 75 (based on original author's trial and error) The look_back variable controls the size of the input vector into the RNN(LSTM).  

Sensitivity analysis options:
1. the timestep of the input vector
2. LSTM unit which is the internal vector size of the cell (memory) and hidden state 

Note that each character is represented by a one-hot vector with the size equals to the character "universe" size that happens to be in the original text.  Keras provides a handy  method for converting character into one-hot:

```python
# separate into input and output
sequences = array(sequences)
X = sequences[:,:-1]	# every column but last column of every row
y = sequences[:,-1]		# last column of every row ; used as labelled data
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)
```

Keras LSTM model construct (single layer LSTM):

```python
# define model
model = Sequential()
# single hidden layer (timestep is one) and 75 cells
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=2)
```

The learned LSTM model is stored in model.h5 file. 

## Text generation based on learned LSTM model

The text generation results are created by the running the saved model.h5.  A lead-off (seed text) phrase is input into the function such that it can start off from a short sequence of text for further prediction. Each newly generated character becomes the new seed for the next round of prediction.

Example seed text for lead-off:

```python
# test start of star wars
print(generate_seq(model, mapping, 10, 'Episode IX tells the story of', 1000))
# test mid-line
print(generate_seq(model, mapping, 10, 'In Episode IX the first order', 1000))
# test not in original
print(generate_seq(model, mapping, 10, 'Skywalker eats like a dogs', 1000))
```


First generated block of text (1000 characters):

*Episode IX tells the story of Enotir pland andized by of Luke's stuens. Qui-Gon senses a stroug travels to Mustafar. On the asteroid Rosi-s, chate PadmÃ©i, and toruses to stack aster tale. The Jedi escape and flee Cloud City on the surface. They rescue new he piss ane eventsace. Palpatine explains that Luke attempues to pilc tares settle. Luke and Leia in pain and fichrorvers Simia's for droid army while PadmÃ© leads the group sis bith hangar of the Senate and droid army while PadmÃ© leads the group sis bith hangar of the Senate and droid army while PadmÃ© leads the group sis bith hangar of the Senate and droid army while PadmÃ© leads the group sis bith hangar of the Senate and droid army while PadmÃ© leads the group sis bith hangar of the Senate and droid army while PadmÃ© leads the group sis bith hangar of the Senate and droid army while PadmÃ© leads the group sis bith hangar of the Senate and droid army while PadmÃ© leads the group sis bith hangar of the Senate and droid army while PadmÃ© leads the group sis bi*


Second generated block of text (characters):

*In Episode IX the first ordersing, Luke Snokukes oftermed and is hon clpeplives Anakbo takes soy trenturr the sturs of his plone in the Falcon. Luker vote pensels to rescue Obi-Wan, and meessing that he will befoins his father, Vader reveals that he is the "chosen one" of the Jedi Council as on the stain to a nearthe. Luke intention of the Emperor tortures Luke with Force lightning. Unwilling to let to vis to the Jedi Council as on the stain to a nearthe. Luke intention of the Emperor tortures Luke with Force lightning. Unwilling to let to vis to the Jedi Council as on the stain to a nearthe. Luke intention of the Emperor tortures Luke with Force lightning. Unwilling to let to vis to the Jedi Council as on the stain to a nearthe. Luke intention of the Emperor tortures Luke with Force lightning. Unwilling to let to vis to the Jedi Council as on the stain to a nearthe. Luke intention of the Emperor tortures Luke with Force lightning. Unwilling to let to vis to the Jedi Council as on the stain to a nearthe. Luke inte*

Third generated block of text (characters)

*Skywalker eats like a dogsous an mentinal from Ahih With by om an the probe alerts the Imperials. Meanwhile, Luke arrives at mid-sind, Luke Senorem, whal reveles Geyeled Luke's father, Vader reveals that he is the "chosen one" of the Jedi Council as on the stain to a nearthe. Luke intention of the Emperor tortures Luke with Force lightning. Unwilling to let to vis to the Jedi Council as on the stain to a nearthe. Luke intention of the Emperor tortures Luke with Force lightning. Unwilling to let to vis to the Jedi Council as on the stain to a nearthe. Luke intention of the Emperor tortures Luke with Force lightning. Unwilling to let to vis to the Jedi Council as on the stain to a nearthe. Luke intention of the Emperor tortures Luke with Force lightning. Unwilling to let to vis to the Jedi Council as on the stain to a nearthe. Luke intention of the Emperor tortures Luke with Force lightning. Unwilling to let to vis to the Jedi Council as on the stain to a nearthe. Luke intention of the Emperor tortures Luke with*

# Observations:

1. The LSTM seems to go into loops once it find a sequence that it is "familiar" with such as the repetitiveness in the first block.
2. Certain characters must have a high significance of what should follow which led the LSTM to produce some humanly readile text.
3. Since this is a character level learning, it is not expected that generated text should follow proper grammer or have sematic meanings (not to the mention the raw text for learning was a relatively small training set.)


# Acknowledgment

Thanks to Dr. Jason Brownlee from [machinelearningmastery.com](https://machinelearningmastery.com/) for providing the base code. 