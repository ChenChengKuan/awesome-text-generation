# awesome-text-generation
## Model
### Reinforcement learning based
   * SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient [[pdf]](https://arxiv.org/abs/1609.05473)[[code (official)]](https://github.com/LantaoYu/SeqGAN)[[code (non-official)]](https://github.com/ChenChengKuan/SeqGAN_tensorflow)
      * Lantao Yu, Weinan Zhang, Jun Wang, Yong Yu *AAAI 2017*
   * Long Text Generation via Adversarial Training with Leaked Information[[pdf]](https://arxiv.org/abs/1709.08624)[[code]](https://github.com/CR-Gjx/LeakGAN)
      * Jiaxian Guo, Sidi Lu, Han Cai, Weinan Zhang, Yong Yu, Jun Wang *AAAI 2018*
   * MaskGAN: Better Text Generation via Filling in the______ [[pdf]](https://arxiv.org/abs/1801.07736)[[code]](https://github.com/tensorflow/models/tree/master/research/maskgan)
      * William Fedus, Ian Goodfellow, Andrew M. Dai *ICLR 2018*
   * Adversarial ranking for language generation [[pdf]](https://arxiv.org/abs/1705.11001)
      * Kevin Lin, Dianqi Li, Xiaodong He, Zhengyou Zhang, Ming-Ting Sun *AAAI 2018*
   
### Autoencoder based
   * Adversarially Regularized Autoencoders for Generating Discrete Structures [[pdf]](https://arxiv.org/abs/1706.04223)[[code]](https://github.com/jakezhaojb/ARAE)
      * Jake Zhao (Junbo), Yoon Kim, Kelly Zhang, Alexander M. Rush, Yann LeCun *ICML 2018*

### VAE based
   * Avoiding Latent Variable Collapse with Generative Skip Models [[pdf]](https://arxiv.org/pdf/1807.04863.pdf)
      * Adji B. Dieng, Yoon Kim, Alexander M. Rush, David M. Blei *ICML 2018 workshop on Theoretical Foundations
and Applications of Deep Generative Models*
   * Variational Attention for Sequence-to-Sequence Models [[pdf]](https://arxiv.org/abs/1712.08207)
      * Hareesh Bahuleyan, Lili Mou, Olga Vechtomova, Pascal Poupart *COLING 2018*
   * A Hybrid Convolutional Variational Autoencoder for Text Generation [[pdf]](https://arxiv.org/abs/1702.02390)
      * Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth *EMNLP 2017*
   * Piecewise Latent Variables for Neural Variational Text Processing [[pdf]](https://www.aclweb.org/anthology/D17-1043)
      * Iulian V. Serban1 and Alexander G. Ororbia II2 and Joelle Pineau3 and Aaron Courville1 *EMNLP 2017*
   * Z-Forcing: Training Stochastic Recurrent Networks [[pdf]](https://arxiv.org/pdf/1711.05411.pdf)
      * Anirudh Goyal, Alessandro Sordoni, Marc-Alexandre Côté, Nan Rosemary Ke, Yoshua Bengio *NIPS 2017*
   * Toward Controlled Generation of Text [[pdf]](https://arxiv.org/abs/1703.00955)[[code]](https://github.com/wiseodd/controlled-text-generation)
      * Zhiting Hu, Zichao Yang, Xiaodan Liang, Ruslan Salakhutdinov, Eric P. Xing *ICML 2017*
   * Improved Variational Autoencoders for Text Modeling using Dilated Convolutions [[pdf]](https://arxiv.org/abs/1702.08139)[[code]](https://github.com/ryokamoi/dcnn_textvae)
      * Zichao Yang, Zhiting Hu, Ruslan Salakhutdinov, Taylor Berg-Kirkpatrick *ICML 2017*
   * Generating Sentences from a Continuous Space [[pdf]](https://arxiv.org/abs/1511.06349)
      * Samuel R. Bowman, Luke Vilnis, Oriol Vinyals, Andrew M. Dai, Rafal Jozefowicz, Samy Bengio *CoNLL 2016*
   
### GAN based
   *  GANS for Sequences of Discrete Elements with the Gumbel-softmax Distribution [[pdf]](https://arxiv.org/abs/1611.04051)
      * Matt J. Kusner, José Miguel Hernández-Lobato *ICLR 2018*
   *  Boundary-Seeking Generative Adversarial Networks [[pdf]](https://arxiv.org/abs/1702.08431)
      * R Devon Hjelm, Athul Paul Jacob, Tong Che, Adam Trischler, Kyunghyun Cho, Yoshua Bengio *ICLR 2018*
   *  Adversarial Feature Matching for Text Generation [[pdf]](https://arxiv.org/abs/1706.03850) [[code]](https://github.com/dreasysnail/textGAN_public)
      * Yizhe Zhang, Zhe Gan, Kai Fan, Zhi Chen, Ricardo Henao, Dinghan Shen, Lawrence Carin *ICML 2017*
   *  Improved Training of Wasserstein GANs [[pdf]](https://arxiv.org/abs/1704.00028)[[code]](https://github.com/igul222/improved_wgan_training)
      * Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville *NIPS 2017*
   *  Sobolev GAN [[pdf]](https://arxiv.org/abs/1711.04894)
      * Youssef Mroueh, Chun-Liang Li, Tom Sercu, Anant Raj, Yu Cheng *ICLR 2018*
      
### Alternative decode objective
   * Learning to Write with Cooperative Discriminators [[pdf]](https://arxiv.org/abs/1805.06087)[[code]](https://github.com/ari-holtzman/l2w)
      * Ari Holtzman, Jan Buys, Maxwell Forbes, Antoine Bosselut, David Golub, Yejin Choi *ACL 2018*
      
### Tool
   *  Texar: A Modularized, Versatile, and Extensible Toolbox for Text Generation [[pdf]](http://www.aclweb.org/anthology/W18-2503)
      * Zhiting Hu, Zichao Yang, Haoran Shi, Bowen Tan, Tiancheng Zhao,Junxian He, Xiaodan Liang, Wentao Wang, Xingjiang Yu, Di Wang, Lianhui Qin, Xuezhe Ma, Hector Liu, Devendra Singh, Wangrong Zhu, Eric P. Xing *ACL 2018*

### Others
   * SentiGAN: Generating Sentimental Texts via Mixture Adversarial Networks
   * Neural Text Generation: Past, Present and Beyond

## Applications
### Reinforcement Learning based/aided text generation
#### Image to text

   * Recurrent Topic-Transition GAN for Visual Paragraph Generation [[pdf]](https://arxiv.org/abs/1703.07022)
      * Xiaodan Liang, Zhiting Hu, Hao Zhang, Chuang Gan, Eric P. Xing *ICCV 2017*
   * Towards Diverse and Natural Image Descriptions via a Conditional GAN [[pdf]](https://arxiv.org/abs/1703.06029)
      * Bo Dai, Sanja Fidler, Raquel Urtasun, Dahua Lin *ICCV 2017*
   * Show, Adapt and Tell: Adversarial Training of Cross-domain Image Captioner [[pdf]](https://arxiv.org/abs/1705.00930)[[code]](https://github.com/tsenghungchen/show-adapt-and-tell)
      * Tseng-Hung Chen, Yuan-Hong Liao, Ching-Yao Chuang, Wan-Ting Hsu, Jianlong Fu, Min Sun *ICCV 2017*
   * Improved Image Captioning via Policy Gradient optimization of SPIDEr [[pdf]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Improved_Image_Captioning_ICCV_2017_paper.pdf)
      * Liu, Siqi; Zhu, Zhenhai; Ye, Ning; Guadarrama, Sergio; Murphy, Kevin *ICCV 2017*

#### Stylistic Text
   *  SentiGAN: Generating Sentimental Texts via Mixture Adversarial Networks [[pdf]](https://www.ijcai.org/proceedings/2018/0618.pdf)
      * Ke Wang, Xiaojun Wan *IJCAI 2018*

#### (Visual) Dialogue
   * Best of Both Worlds: Transferring Knowledge from Discriminative Learning to a Generative Visual Dialog Model [[pdf]](https://papers.nips.cc/paper/6635-best-of-both-worlds-transferring-knowledge-from-discriminative-learning-to-a-generative-visual-dialog-model.pdf)
   * Are You Talking to Me? Reasoned Visual Dialog Generation through Adversarial Learning [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Are_You_Talking_CVPR_2018_paper.pdf)
      * Qi Wu, Peng Wang, Chunhua Shen, Ian Reid, Anton van den Hengel *CVPR 2018*
   * Adversarial Learning for Neural Dialogue Generation [[pdf]](https://arxiv.org/pdf/1701.06547.pdf)
      * Jiwei Li, Will Monroe, Tianlin Shi, Sébastien Jean, Alan Ritter, Dan Jurafsky *EMNLP 2017*
   * Learning cooperative visual dialog agents with deep reinforcement learning [[pdf]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Das_Learning_Cooperative_Visual_ICCV_2017_paper.pdf)
      * Abhishek Das, Satwik Kottur, José M.F. Moura, Stefan Lee *ICCV 2017*
, Dhruv Batra1,4
   * Polite Dialogue Generation Without Parallel Data [[pdf]](https://arxiv.org/abs/1805.03162)
      * Tong Niu, Mohit Bansal *TACL 2018*
#### Other
   *  Generating Reasonable and Diversified Story Ending Using Sequence to Sequence Model with Adversarial Training [[pdf]](http://www.aclweb.org/anthology/C18-1088)
      * Zhongyang Li, Xiao Ding and Ting Liu *COLING 2018*
      
### VAE based
### (Visual) Dialogue
   * Unsupervised Discrete Sentence Representation Learning for Interpretable Neural Dialog Generation [[pdf]](https://arxiv.org/pdf/1804.08069.pdf)
      * Tiancheng Zhao, Kyusong Lee and Maxine Eskenazi *ACL 2018*
   * Improving Variational Encoder-Decoders in Dialogue Generation [[pdf]](https://arxiv.org/pdf/1802.02032.pdf)
      * Xiaoyu Shen, Hui Su, Shuzi Niu and Vera Demberg *AAAI 2018*
   * A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues [[pdf]](https://arxiv.org/abs/1605.06069)
      * Iulian Vlad Serban, Alessandro Sordoni, Ryan Lowe, Laurent Charlin, Joelle Pineau, Aaron Courville, Yoshua Bengio *AAAI 2017*
   * Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders [[pdf]](https://arxiv.org/abs/1703.10960)
      * Tiancheng Zhao, Ran Zhao and Maxine Eskenazi *ACL 2017*
### GAN based (Adversarial Learining)
#### Image to text
   * Speaking the Same Language: Matching Machine to Human Captions by Adversarial Training [[pdf]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Shetty_Speaking_the_Same_ICCV_2017_paper.pdf)
      * Rakshith Shetty, Marcus Rohrbach, Lisa Anne Hendricks, Mario Fritz1 Bernt Schiele *ICCV 2017*
#### Stylistic Text
   * Style Transfer Through Back-Translation [[pdf]](https://arxiv.org/pdf/1804.09000.pdf)
      * Shrimai Prabhumoye, Yulia Tsvetkov, Ruslan Salakhutdinov, Alan W Black *ACL 2018*
   * Style Transfer in Text: Exploration and Evaluation [[pdf]](https://arxiv.org/pdf/1711.06861.pdf)
      * Zhenxin Fu, Xiaoye Tan, Nanyun Peng, Dongyan Zhao, Rui Yan *AAAI 2018*
   * Style Transfer from Non-Parallel Text by Cross-Alignment [[pdf]](https://arxiv.org/abs/1705.09655)
      * Tianxiao Shen, Tao Lei, Regina Barzilay, Tommi Jaakkola *NIPS 2017*
      
#### Other
   * Conditional Generative Adversarial Networks for Commonsense Machine Comprehension [[pdf]](https://www.ijcai.org/proceedings/2017/0576.pdf)
      * Bingning Wang, Kang Liu, Jun Zhao *IJCAI 2017*
   * Unsuprervised Cipher Cracking Using Discrete GANS [[pdf]](https://openreview.net/pdf?id=BkeqO7x0-)
      * Aidan N. Gomez, Sicong Huang, Ivan Zhang, Bryan M. Li, Muhammad Osama, Łukasz Kaiser *ICLR 2018*
      
   * Multimodal Storytelling via Generative Adversarial Imitation Learning [[pdf]](https://www.ijcai.org/proceedings/2017/0554.pdf)
      * Zhiqian Chen, Xuchao Zhang, Arnold P. Boedihardjo, Jing Dai, Chang-Tien Lu1, *IJCAI 2017*
      
 ### Other
   * Compositional Obverter Communication Learning From Raw Visual Input [[pdf]](https://arxiv.org/abs/1804.02341)
      * Edward Choi, Angeliki Lazaridou, Nando de Freitas *ICLR 2018*
   * Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory [[pdf]](https://arxiv.org/abs/1704.01074)
      * Hao Zhou, Minlie Huang, Tianyang Zhang, Xiaoyan Zhu, Bing Liu *AAAI 2018*
