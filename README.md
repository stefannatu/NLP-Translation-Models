# NLP-Translation-Models
Experimenting with different architectures for neural machine translation


### I'm experimenting with different deep learning architectures for Neural Machine Translation on a small dataset (English to Romanian from 'http://www.manythings.org/anki/ron-eng.zip' which is a small dataset of roughly 8000 English and Romanian phrases. To train a much better model at scale, recommend using the WMT 14 dataset which was used in Gehring *et al* https://arxiv.org/pdf/1705.03122.pdf from Facebook AI Research. 


### The ultimate goal of this repo is to understand more recent literature pointing to convolutional neural networks outperforming RNNs and Bi-Directional LSTMs on sequence learning tasks. I use the small dataset for prototyping and the larger dataset for final training of the model. The .py files contain the original code for the models. The code is modular, and written usig classes which can be simply lifted as is and put into other models -- for example the encoder-Decoder model class definitions can also be used to train a text generation model (see the Seq_2_Seq repo)

### Starter Model -- The benchmark model I use is an encoder-decoder model which closely follows the Tensorflow tutorial on the topic.

### Stay tuned for more developments. 
