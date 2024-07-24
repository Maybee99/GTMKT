# GTMKT
GAT-Transformer Memory Networks for Forgetting Knowledge Tracing

## Overall Architecture
![图片](./GTMKT/GTMKT.png)

## Dataset
[Junyi](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198)

[Assist09](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data)

[Static2011](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507)

## Installation

```bash
git clone https://github.com/Maybee99/GTMKT.git
cd GTMKT
pip3 instll -r requirements
```

```
pip3 install torch==1.9.1 torch_geometric==2.0.1 torch_sparse==0.6.12 torch_scatter==2.0.9
```
## Usage

### Train

Train GTMKT with CL loss:

```bash
python3 ./train.py -m GTMKT -d [junyi, assist09,,statics] -bs 1 -e 20 -lr 0.003
```

For more options, run:

```bash
python3 ./train.py -h
```