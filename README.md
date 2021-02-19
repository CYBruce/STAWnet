# Spatial-Temporal Attention Wavenet: a deep learning framework for traffic prediction considering spatial-temporal dependencies 

## Reference

This repository contains the *source code* of the following research paper:

- C.Y. Tian, W.K.V. Chan, Spatial-Temporal Attention Wavenet: a deep learning framework for traffic prediction considering spatial-temporal dependencies,IET Intelligent Traffic Systems

## Introduction

Traffic prediction on road networks is highly challenging due to the complexity of traffic systems and is a crucial task in successful intelligent traffic system applications. Existing approaches mostly capture the static spatial dependency relying on the prior knowledge of the graph structure. However, the spatial dependency can be dynamic, and sometimes the physical structure may not reflect the genuine relationship between roads. To better capture the complex spatial-temporal dependencies and forecast traffic conditions on road networks, we propose a multi-step prediction model named Spatial-Temporal Attention Wavenet (STAWnet). Temporal convolution is applied to handle long time sequences, and the dynamic spatial dependencies between different nodes can be captured using the self-attention network. Different from existing models, STAWnet does not need prior knowledge of the graph by developing a self-learned node embedding. These components are integrated into an end-to-end framework. The experimental results on three public traffic prediction datasets (METR-LA, PEMS-BAY, and PEMS07) demonstrate effectiveness. In particular, in the 1 hour ahead prediction, STAWnet outperforms state-of-the-art methods with no prior knowledge of the network.

## Data Preparation

Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN). Process raw data

```
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

## Train Commands

```
python train.py
```

## Contact

If you have any question, please feel free to send an E-mail to tiancy19@mails.tsinghua.edu.cn.
