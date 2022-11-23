## Bert  中文文本分类
参考code: 
- https://github.com/CyberZHG/keras-bert
- https://github.com/google-research/bert
- https://github.com/bojone/bert4keras

![image](https://user-images.githubusercontent.com/36963108/203489367-de15addb-addf-447b-9273-64b9ba4f8d41.png)


## 运行环境
- python3.6
- tensorflow1.15


## 任务 判断文本相似性
#### 数据说明 
数据来自己：https://github.com/wslc1314/atec_nlp_sim_update/tree/master/data/atec

数据的标签为1表示两句话相匹配，标签为0表示两句话不匹配

- atec_nlp_sim_train_0.6.csv和atec_nlp_sim_test_0.4.csv格式相同 
- 各列分别为"index query1 query2 label"，"\t"分隔。来自网络上蚂蚁金服的公开数据集，是判断文本相似性的数据  
- 程序运行中只选择了部分数据(train3000,val500,test100)  


#### 训练 与 预测

```buildoutcfg
python run_classifier_mian.py.py --task_name=MAYI --do_train=true --do_eval=true --data_dir=./ --vocab_file=./chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=./chinese_L-12_H-768_A-12/bert_
config.json --init_checkpoint=./chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./output/
 
python run_classifier_mian.py.py --task_name=MAYI --do_train=false --do_eval=false --do_predict=true --data_dir=./ --vocab_file=./chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=./chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --output_dir=./output/
  
```

 

# 参考资料

Bert文本分类(基于keras-bert实现)：https://blog.csdn.net/asialee_bird/article/details/102747435 

Bert进行二分类：
- https://blog.csdn.net/bull521/article/details/105044528/  
- https://github.com/Hejp5665/bert_keras_nlp  
- https://zhuanlan.zhihu.com/p/61671334 
- https://cloud.tencent.com/developer/article/1601710

基于keras-bert实现训练，保存，加载，预测单个文本:https://blog.csdn.net/hejp_123/article/details/105432539

Bert_Classification：https://github.com/morenjiujiu/Chinese_Bert_Classification

数据处理参考：
- https://blog.csdn.net/yingdajun/article/details/117985048
- https://blog.csdn.net/Rock_y/article/details/107325750
- https://zhuanlan.zhihu.com/p/145192287
- https://www.cnblogs.com/hcxss/p/15894028.html


