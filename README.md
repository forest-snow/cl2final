# cl2final

Our project has 3 components: n-gram model, topic model, and RNN.

## Requirements
- [sLDA toolkit](https://github.com/dongwookim-ml/python-topic-model)
- [PyTorch](https://github.com/pytorch/pytorch)

## Run
### N-gram Model
To run the n-gram model, run `python KL_div.py`. You may need to change:
* path in KL_div.py: where the reddit dataset is located

### Topic Model
To run the topic model, run `python slda.py`. You may need to change:
* pathfile in extract.py: where the reddit dataset is located
* output_file in slda.py: where the output topic vectors will be saved

### Recurrent Neural Networks (RNN)
To run the RNN model, run `python train.py`. You may need to change:
* path in train.py: where the reddit dataset is located
* pickle_file in train.py: where the post representations are stored
* model_file in train.py: where RNN model will be saved

