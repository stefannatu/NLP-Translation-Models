# NLP-Translation-Models
Experimenting with different architectures for neural machine translation


### I'm experimenting with different deep learning architectures for Neural Machine Translation on a small dataset (English to Romanian from 'http://www.manythings.org/anki/ron-eng.zip' which is a small dataset of roughly 8000 English and Romanian phrases. To train a much better model at scale, recommend using the WMT 14 dataset which was used in Gehring *et al* https://arxiv.org/pdf/1705.03122.pdf from Facebook AI Research, or the EuroParl dataset. Beware that this dataset is huge and training will take a while. 

### The ultimate goal of this repo is to understand more recent literature pointing to convolutional neural networks outperforming RNNs and Bi-Directional LSTMs on sequence learning tasks. I use the small dataset for prototyping and the larger dataset for final training of the model. The .py files contain the original code for the models. The code is modular, and written usig classes which can be simply lifted as is and put into other models -- for example the encoder-Decoder model class definitions can also be used to train a text generation model (see the Seq_2_Seq repo)

### Starter Model -- The benchmark model I use is an encoder-decoder model trained with Bahnadau attention, which closely follows the Tensorflow tutorial on the topic. As you can see from Translation.ipynb, the model is trained on a small set of data, and for only few epochs and it doesn't perform as well as it should. Training this model on a larger dataset is left as future work.

### Stay tuned for more developments. Currently, I am working on Convolutional-Translation models which have appeared in recent literature and been shown to outperform RNN's and LSTMs on long sequences. I am training this model on the much larger EuroParl dataset. An updated notebook containing results will be uploaded soon.

