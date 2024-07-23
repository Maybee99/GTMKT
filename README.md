# GTMKT
GAT-Transformer Memory Networks for Forgetting Knowledge Tracing

## Overall Architecture
![图片](https://user-images.githubusercontent.com/77867386/165916825-9c2135cc-d83b-43b4-82bb-c059a49af7e1.png)

## Dataset
We evaluate our method on three benchmark datasets for knowledge tracing, i.e., ASSIST09, ASSIST17, and EdNet.
In addition to the ASSIST17 dataset provided in the code, the ASSIST09 and EdNet datasets which mentioned in the paper are in the Google Drive, which you can download with this [link](https://drive.google.com/file/d/1ItqFv0fH6ibTotmflFNeAX0d7PdMaF7B/view?usp=sharing](https://edudata.readthedocs.io/en/latest/).
### Try using your own dataset!

You can also use your own data set, but note that besides the four row data set, you also need to build the incidence matrix of question and concept of the dataset and put it in the `/Dataset/H` folder.
## Models

 - `/KnowledgeTracing/model/Model.py`:end-to-end prediction framework;
 -  `/KnowledgeTracing/hgnn_models`:the module of Concept Association HyperGraph(CAHG);
 -  `KnowledgeTracing/DirectedGCN`:the module of Directed Transition Graph (DTG);
 - `/KnowledgeTracing/data/`:reading and processing datasets;
 - `/KnowledgeTracing/evaluation/eval.py`:Calculate losses and performance;

## Setup

To run this code you need the following:

    a machine with GPUs
    python3
    numpy, pandas, scipy, scikit-learn and torch packages:
```
pip3 install torch==1.7.0 numpy==1.21.2 pandas==1.4.1 scipy==1.7.3 scikit-learn==1.0.2 tqdm==4.26.3 
```
## Hyperparameter Settings
`/KnowledgeTracing/Constant/Constants.py` is specially used to set super parameters, all parameters of the whole model can be set in this file.

> MAX_STEP = 50 
> BATCH_SIZE = 128 
> LR = 0.001 
> EPOCH = 20 
> EMB = 256 
> HIDDEN = 128 
> kd_loss = 5.00E-06

## Save Log

If you need to save the log, create a `log` folder under the evaluation folder.
There are trained models in the model folder, which can be directly run `KTtest()` in  `run.py`  . 
Of course, you can also train a new models, just  run  `KTtrain()` in `run.py`

## Contact us
If you have any questions, please contact iext48@163.com.

