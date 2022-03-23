[resnet 代码详解](https://zhuanlan.zhihu.com/p/77899090)
[resnet18 网络详细结构](https://zhuanlan.zhihu.com/p/353185272)
[超级详细的ResNet代码解读（Pytorch）](https://zhuanlan.zhihu.com/p/474790387)

[pytorch 代码结构（陈云）](https://zhuanlan.zhihu.com/p/29024978)
[PyTorch 实践指南(陈云)，与上面知乎的文章相对应](https://github.com/chenyuntc/pytorch-best-practice)
[pytorch 项目规范（depvac）](https://blog.csdn.net/qq_44554428/article/details/121506485?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3.pc_relevant_default&utm_relevant_index=6)
[详细介绍和说明](https://blog.csdn.net/teeyohuang/article/details/79210525)

github代码参考
https://github.com/drgripa1/resnet-cifar10：分别写成了train.py,test.py文件，整体结构与我的结构较为类似
https://github.com/shruti-bt/cifar10-models：使用了argparser,train.py,main.py中参考了很多内容
https://github.com/zhulf0804/Resnet-Pytorch：中使用了tensorboard,使用SummaryWriter写日志
https://github.com/DanielWong0623/ResNet-18-on-CIFAR-10-using-PyTorch：惨开了具体的模型结构，但是按论文中用cifar有特殊的结构（20层，不是像imagenet使用的224*224图片输入，这个代码实现的是用imagenet的结构在cifar上运行），与[resnet 代码详解](https://zhuanlan.zhihu.com/p/77899090)相互参考