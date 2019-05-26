# Pytorch_Medical_Segmention_Template
### 本工程文件实现了如下功能：
1.基于Unet的单类分割  
2.自动实现n折交叉验证  
3.损失函数为Dice+BCE  
4.优化器为SGD，ploy学习策略  
5.自动保存n折的checkpoint文件  
6.自动保存n折的tensorboard log日志，支持前后多次实验可视化对比 只需将UNet文件夹复制后重命名为：“UNet_修改内容”即可，在此基础上修改  

### 想用此工程你需要做的：
#### 第一步：在Dataset文件夹下创建固定格式的数据文件夹（以Linear_lesion数据为例）：

##### ─Linear_lesion
    ├─img
    │  ├─f1
    │  ├─f2
    │  ├─f3
    │  ├─f4
    │  └─f5
    └─mask
        ├─f1
        ├─f2
        ├─f3
        ├─f4
        └─f5
### 第二步：
根据自己的情况修改Pytorch_Project_template\Linear_lesion_Code\UNet\utils\config.py文件  
### 第三步：
运行Pytorch_Project_template\Linear_lesion_Code\UNet的train.py文件  
