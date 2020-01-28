# A Paper List for Neural-Network-Compression
This is a paper list for neural network compression techniques such as quantization, pruning, and distillation. Most of the papers in the list are about quantization of CNNs. Only official codes are crosslinked.

# Paper List
format: (**[Nickname]**) Paper title, published @, paper link, (official code (if provided))

## Quantization
- **[BNN]** Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, arXiv 2016, [[paper]](https://arxiv.org/abs/1602.02830), [[code(Theano)]](https://github.com/MatthieuCourbariaux/BinaryNet) [[code(Torch-7)]](https://github.com/itayhubara/BinaryNet), [[code(Pytorch)]](https://github.com/itayhubara/BinaryNet.pytorch), [[code(Tensorflow)]](https://github.com/itayhubara/BinaryNet.tf)

- **[XNOR-NET]** XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks, ECCV 2016, [[paper]](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_32), [[code(Torch-7)]](https://github.com/allenai/XNOR-Net)
- **[DoReFa]** DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients, arXiv 2016, [[paper]](https://arxiv.org/abs/1606.06160), [[code(Tensorflow)]](https://github.com/tensorpack/tensorpack/tree/master/examples/DoReFa-Net)
- **[HWGQ]** Deep Learning With Low Precision by Half-Wave Gaussian Quantization, CVPR 2017, [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/html/Cai_Deep_Learning_With_CVPR_2017_paper.html), [[code(caffe)]](https://github.com/zhaoweicai/hwgq)
- **[TWN]** Ternary Weight Networks, NIPS workshop 2016, [[paper]](https://arxiv.org/abs/1605.04711), [[code(caffe)]](https://github.com/fengfu-chris/caffe-twns)
- **[TTQ]** Trained Ternary Quantization, ICLR 2017, [[paper]](https://openreview.net/forum?id=S1_pAu9xl&noteId=S1_pAu9xl), [[code(Tensorflow)]](https://github.com/czhu95/ternarynet)
- How to train a compact binary neural network with high accuracy?, AAAI 2017, [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14619)
- **[ABC-Net]** Towards Accurate Binary Convolutional Neural Network, NIPS 2017, [[paper]](http://papers.nips.cc/paper/6638-towards-accurate-binary-convolutional-neural-network)
- **[WEQ]** Weighted-Entropy-Based Quantization for Deep Neural Networks, CVPR 2017, [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/html/Park_Weighted-Entropy-Based_Quantization_for_CVPR_2017_paper.html), [[code(caffe)]](https://github.com/EunhyeokPark/script_for_WQ)
- **[Network Sketching]** Network Sketching: Exploiting Binary Structure in Deep CNNs, CVPR 2017, [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/html/Guo_Network_Sketching_Exploiting_CVPR_2017_paper.html)
- **[WRPN]** WRPN: Wide Reduced-Precision Networks, ICLR 2018, [[paper]](https://openreview.net/forum?id=B1ZvaaeAZ&noteId=B1ZvaaeAZ)
- **[PACT]** PACT: Parameterized Clipping Activation for Quantized Neural Networks, arXiv 2018, [[paper]](https://arxiv.org/abs/1805.06085)
- **[Bi-Real-Net]** Bi-Real Net: Enhancing the Performance of 1-bit CNNs with Improved Representational Capability and Advanced Training Algorithm, ECCV 2018, [[paper]](http://openaccess.thecvf.com/content_ECCV_2018/html/zechun_liu_Bi-Real_Net_Enhancing_ECCV_2018_paper.html), [[code(caffe)]](https://github.com/liuzechun/Bi-Real-net)
- **[SYQ]** SYQ: Learning Symmetric Quantization For Efficient Deep Neural Networks, CVPR 2018, [[paper]](http://openaccess.thecvf.com/content_cvpr_2018/html/Faraone_SYQ_Learning_Symmetric_CVPR_2018_paper.html), [[code(Tensorflow)]](https://github.com/julianfaraone/SYQ)
- Towards Effective Low-Bitwidth Convolutional Neural Networks, CVPR 2018, [[paper]](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhuang_Towards_Effective_Low-Bitwidth_CVPR_2018_paper.html), [[code(Pytorch)]](https://github.com/nowgood/QuantizeCNNModel)
- **[TSQ]** Two-Step Quantization for Low-bit Neural Networks, CVPR 2018, [[paper]](http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Two-Step_Quantization_for_CVPR_2018_paper.html)
- **[LQ-Net]** LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks, ECCV 2018, [[paper]](http://openaccess.thecvf.com/content_ECCV_2018/html/Dongqing_Zhang_Optimized_Quantization_for_ECCV_2018_paper.html), [[code(Tensorflow)]](https://github.com/microsoft/LQ-Nets)
- Alternating Multi-bit Quantization for Recurrent Neural Networks, ICLR 2018, [[paper]](https://openreview.net/forum?id=S19dR9x0b)
- **[NICE]** NICE: Noise Injection and Clamping Estimation for Neural Network Quantization, arXiv 2018, [[paper]](https://arxiv.org/abs/1810.00162)
- **[Continuous Binarization]** True Gradient-Based Training of Deep Binary Activated Neural Networks Via Continuous Binarization, ICASSP 2018, [[paper]](https://ieeexplore.ieee.org/abstract/document/8461456/)
- **[MCDQ]** Model compression via distillation and quantization, ICLR 2018, [[paper]](https://openreview.net/forum?id=S1XolQbRW), [[code(Pytorch)]](https://github.com/antspy/quantized_distillation)
- **[Apprentice]** Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy, ICLR 2018, [[paper]](https://openreview.net/forum?id=B1ae1lZRb&noteId=B1ae1lZRb)
- **[Integer]** Training and Inference with Integers in Deep Neural Networks, ICLR 2018, [[paper]](https://openreview.net/forum?id=HJGXzmspb), [[code(Tensorflow)]](https://github.com/boluoweifenda/WAGE)
- Heterogeneous Bitwidth Binarization in Convolutional Neural Networks, NIPS 2018, [[paper]](http://papers.nips.cc/paper/7656-heterogeneous-bitwidth-binarization-in-convolutional-neural-networks)
- An empirical study of Binary Neural Networks' Optimisation, ICLR 2019, [[paper]](https://openreview.net/forum?id=rJfUCoR5KX)
- **[PACT-SAWB]** Accurate and Efficient 2-bit Quantized Neural Networks, SysML 2019, [[paper]](https://www.sysml.cc/doc/2019/168.pdf)
- **[QIL]** Learning to Quantize Deep Networks by Optimizing Quantization Intervals With Task Loss, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Jung_Learning_to_Quantize_Deep_Networks_by_Optimizing_Quantization_Intervals_With_CVPR_2019_paper.html)
- **[Group-Net]** Structured Binary Neural Networks for Accurate Image Classification and Semantic Segmentation, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhuang_Structured_Binary_Neural_Networks_for_Accurate_Image_Classification_and_Semantic_CVPR_2019_paper.html)
- **[ProxQuant]** ProxQuant: Quantized Neural Networks via Proximal Operators, ICLR 2019, [[paper]](https://openreview.net/forum?id=HyzMyhCcK7), [[code(Pytorch)]](https://github.com/allenbai01/ProxQuant)
- **[CBCN]** Circulant Binary Convolutional Networks: Enhancing the Performance of 1-Bit DCNNs With Circulant Back Propagation, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Circulant_Binary_Convolutional_Networks_Enhancing_the_Performance_of_1-Bit_DCNNs_CVPR_2019_paper.html)
- **[CI-BCNN]** Learning Channel-Wise Interactions for Binary Convolutional Neural Networks, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Learning_Channel-Wise_Interactions_for_Binary_Convolutional_Neural_Networks_CVPR_2019_paper.html)
- **[QN]** Quantization Networks, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Yang_Quantization_Networks_CVPR_2019_paper.html)
- **[BNN+]** BNN+: Improved Binary Network Training, arXiv 2019, [[paper]](https://arxiv.org/abs/1812.11800)
- **[DistributionLoss]** Regularizing Activation Distribution for Training Binarized Deep Networks, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Regularizing_Activation_Distribution_for_Training_Binarized_Deep_Networks_CVPR_2019_paper.html), [[code(Pytorch)]](https://github.com/ruizhoud/DistributionLoss)
- A Main/Subsidiary Network Framework for Simplifying Binary Neural Networks, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Xu_A_MainSubsidiary_Network_Framework_for_Simplifying_Binary_Neural_Networks_CVPR_2019_paper.html)
- Matrix and tensor decompositions for training binary neural networks, arXiv 2019, [[paper]](https://arxiv.org/abs/1904.07852)
- **[BENN]** Binary Ensemble Neural Network: More Bits per Network or More Networks per Bit?, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Binary_Ensemble_Neural_Network_More_Bits_per_Network_or_More_CVPR_2019_paper.html), [[code(Pytorch)]](https://github.com/XinDongol/BENN-PyTorch)
- Back to Simplicity: How to Train Accurate BNNs from Scratch?, arXiv 2019, [[paper]](https://arxiv.org/abs/1906.08637), [[code(MXNet)]](https://github.com/hpi-xnor/BMXNet-v2)
- And the Bit Goes Down: Revisiting the Quantization of Neural Networks, arxiv 2019, [[paper]](https://arxiv.org/abs/1907.05686), [[code(Pytorch)]](https://github.com/facebookresearch/kill-the-bits)
- **[BinaryDuo]** BinaryDuo: Reducing Gradient Mismatch in Binary Activation Network by Coupling Binary Activations, ICLR 2020, [[paper]](https://openreview.net/forum?id=r1x0lxrFPS), [[code(Torch-7)]](https://github.com/Hyungjun-K1m/BinaryDuo)
- **[RtB]** Training Binary Neural Networks with Real-to-Binary Convolutions, ICLR 2020, [[paper]](https://openreview.net/forum?id=BJg4NgBKvH)
- **[LLSQ]** Linear Symmetric Quantization of Neural Networks for Low-precision Integer Hardware, ICLR 2020, [[paper]](https://openreview.net/pdf?id=H1lBj2VFPS), [[code(Pytorch]](https://anonymous.4open.science/r/c05a5b6a-1d0c-4201-926f-e7b52034f7a5/)
- **[AutoQ]** AutoQ: Automated Kernel-wise Neural Network Quantization, ICLR 2020, [[paper]](https://openreview.net/pdf?id=rygfnn4twS)
- **[APOT]** Additive Powers-of-two Quantization: An Efficient Non-uniform Discretization for Neural Networks, ICLR 2020, [[paper]](https://openreview.net/pdf?id=BkgXT24tDS), [[code(Pytorch)]](https://github.com/yhhhli/APoT_Quantization)
- **[LSQ]** Learned Step Quantization, ICLR 2020, [[paper]](https://openreview.net/pdf?id=rkgO66VKDS)
- **[MetaQuant]** MetaQuant: Learning to Quantize by Learning to Penetrate Non-differentiable Quantization, NeurIPS 2019, [[paper]](https://papers.nips.cc/paper/8647-metaquant-learning-to-quantize-by-learning-to-penetrate-non-differentiable-quantization.pdf), [[code(Pytorch)]](https://github.com/csyhhu/MetaQuant)

## Pruning 
To be updated.

## Distilation
To be updated.

## Efficient Model
- **[MobileNet]** MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, arXiv 2017, [[paper]](https://arxiv.org/abs/1704.04861)
- **[MobileNet v2]** MobileNetV2: Inverted Residuals and Linear Bottlenecks, CVPR 2018, [[paper]](http://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html)
- **[MobileNet v3]** Searching for MobileNetV3, arXiv 2019, [[paper]](https://arxiv.org/abs/1905.02244)
- **[Xception]** Xception: Deep Learning With Depthwise Separable Convolutions, CVPR 2017, [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html)
- **[ShuffleNet]** ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices, CVPR 2018, [[paper]](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.html)
- **[ShuffleNet v2]** ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design, ECCV 2018, [[paper]](http://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html)
- **[SqueezeNet]** SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size, arXiv 2016, [[paper]](https://arxiv.org/abs/1602.07360)
- **[StrassenNets]** StrassenNets: Deep Learning with a Multiplication Budget, ICML 2018, [[paper]](http://proceedings.mlr.press/v80/tschannen18a.html)
- **[SlimmableNet]** Slimmable Neural Networks, ICLR 2019, [[paper]](https://openreview.net/forum?id=H1gMCsAqY7), [[code(Pytorch)]](https://github.com/JiahuiYu/slimmable_networks)
- **[ChannelNets]** ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions, NIPS 2018, [[paper]](http://papers.nips.cc/paper/7766-channelnets-compact-and-efficient-convolutional-neural-networks-via-channel-wise-convolutions), [[code(Tensorflow)]](https://github.com/HongyangGao/ChannelNets)
- **[Shift]** Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions, CVPR 2018, [[paper]](http://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Shift_A_Zero_CVPR_2018_paper.html), [[code(Pytorch)]](https://github.com/alvinwan/shiftresnet-cifar)
- **[FE-Net]** All You Need is a Few Shifts: Designing Efficient Convolutional Neural Networks for Image Classification, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Chen_All_You_Need_Is_a_Few_Shifts_Designing_Efficient_Convolutional_CVPR_2019_paper.html)
- **[EfficientNet]** EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, ICML 2019, [[paper]](http://proceedings.mlr.press/v97/tan19a.html)


# Copyright 
By Hyungjun Kim (hyungjunkim94@gmail.com) from Pohang University of Science and Technology (POSTECH).  
