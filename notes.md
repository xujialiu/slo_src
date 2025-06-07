
SLO_20250213_1302
改进:
1. util/datasets.py: 修复了val dataset使用的实际上是train dataset的问题
2. engine_finetune.py: 
    1. 修复了roc_auc_score传入参数的问题
    2. 每次val后保存结果到metrics的results_`epoch`.xlsx

SLO_20250213_1457
改进:
1. engine_finetune.py: 
    1. 增加loss值记录
    2. 增加保存cm到metrics文件夹
2. traim_new.py: 
    1. 改为val和test的batch_size为1, 方便保存结果
    2. 增加了类别权重的设置


