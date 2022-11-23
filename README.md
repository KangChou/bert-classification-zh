## Bert  中文文本分类

## 运行环境
- python3.6
- tensorflow1.15


## 任务 判断文本相似性
#### 数据说明 
- atec_nlp_sim_train_0.6.csv和atec_nlp_sim_test_0.4.csv格式相同 \
- 各列分别为"index query1 query2 label"，"\t"分隔。来自网络上蚂蚁金服的公开数据集，是判断文本相似性的数据  
- 程序运行中只选择了部分数据(train3000,val500,test100)  


#### 训练 与 预测

```buildoutcfg
python run_classifier_mian.py.py --task_name=MAYI --do_train=true --do_eval=true --data_dir=./ --vocab_file=./chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=./chinese_L-12_H-768_A-12/bert_
config.json --init_checkpoint=./chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./output/
 
python run_classifier_mian.py.py --task_name=MAYI --do_train=false --do_eval=false --do_predict=true --data_dir=./ --vocab_file=./chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=./chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --output_dir=./output/
  
```

 

