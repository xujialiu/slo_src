SLO_20250213_1614
script:
```python
nohup python train_new.py \
--data_path "/data_A/xujialiu/datasets/daytona" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250211.csv" \
--input_size 448 \
--device "cuda:0" \
--batch_size 8 > ./logs/slo_`date +%Y%m%d_%H%M`.log 2>&1 &
```
aim:
1. 将该训练结果作为baseline, 用于对比其他模型的效果

conclusion:
1. 大约在30个epoch后, loss开始增加, 需要缩小学习率或提前结束训练

----------

slo_20250214_1053
```python
nohup python train_new.py \
--data_path "/data_A/xujialiu/datasets/daytona" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250211.csv" \
--input_size 448 \
--blr 1e-3 \
--device "cuda:0" \
--batch_size 8 > ./logs/slo_`date +%Y%m%d_%H%M`.log 2>&1 &
```
aim:
1. 缩小学习率, 看看效果如何


conclusion:
1. 缩小学习率后, 效果不如之前, 可能需要更多epochs

----------

SLO_20250215_1034
script:
```python
nohup python train_new.py \
--data_path "/data_A/xujialiu/datasets/daytona" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250211.csv" \
--input_size 896 \
--device "cuda:0" \
--batch_size 1 > ./logs/slo_`date +%Y%m%d_%H%M`.log 2>&1 &
```

SLO_20250215_1034
```python
nohup python train_new.py \
--data_path "/data_A/xujialiu/datasets/daytona" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250211.csv" \
--input_size 448 \
--device "cuda:0" \
--batch_size 1 > ./logs/slo_`date +%Y%m%d_%H%M`.log 2>&1 &
```

aim:
1. 查看增加分辨率, 是否能够提高效果
