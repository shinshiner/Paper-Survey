# Content
*   [Machine Learning](https://github.com/shinshiner/Paper-Survey#machine-learning)
    *   [Neural Network](https://github.com/shinshiner/Paper-Survey#neural-network)
    *   [Optimizer](https://github.com/shinshiner/Paper-Survey#optimizer)
*   [Computer Vision](https://github.com/shinshiner/Paper-Survey#computer-vision)
    *   [Network Architecture](https://github.com/shinshiner/Paper-Survey#network-architecture)
    *   [2D Object Detection](https://github.com/shinshiner/Paper-Survey#2d-object-detection)
        *   [Algorithm](https://github.com/shinshiner/Paper-Survey#algorithm)
    *   [2D Segmentation](https://github.com/shinshiner/Paper-Survey#2d-segmentation)
        *   [Algorithm](https://github.com/shinshiner/Paper-Survey#algorithm-1)
    *   [3D Segmentation](https://github.com/shinshiner/Paper-Survey#3d-segmentation)
        *   [Algorithm](https://github.com/shinshiner/Paper-Survey#algorithm-2)
*   [Reinforcement Learning](https://github.com/shinshiner/Paper-Survey#reinforcement-learning)
    *   [Environment](https://github.com/shinshiner/Paper-Survey#environment)
    *   [Algorithm](https://github.com/shinshiner/Paper-Survey#algorithm-3)
*   [Robot](https://github.com/shinshiner/Paper-Survey#robot)
    *   [Grasping](https://github.com/shinshiner/Paper-Survey#grasping)
        *   [Grasping Unknown Objects](https://github.com/shinshiner/Paper-Survey#grasping-unknown-objects)
        *   [Grasping in Cluttered Environment](https://github.com/shinshiner/Paper-Survey#grasping-in-cluttered-environment)
        *   [Grasping via Segmentation](https://github.com/shinshiner/Paper-Survey#grasping-via-segmentation)
        *   [Grasping Points Selection](https://github.com/shinshiner/Paper-Survey#grasping-points-selection)
    *   [Active Perception](https://github.com/shinshiner/Paper-Survey#active-perception)

# Machine Learning

## Neural Network

* [Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf) (**ICML** 2010)

## Optimizer

* [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (**ICLR** 2015)

# Computer Vision

## Network Architecture

* 【VGG16】[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (**arxiv** 2014)

* 【ResNet】[Deep Residual Learning for Image Recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (**openaccess.thecvf** 2016)

## 2D Object Detection

### Algorithm

* 【R-CNN】[Rich feature hierarchies for accurate object detection and semantic segmentation](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf?spm=5176.100239.blogcont55892.8.pm8zm1&file=Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) (**CVPR** 2014)

* [Fast R-CNN](http://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf) (**ICCV** 2015)

* [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf) (**NIPS** 2015)

* 【YOLO】[You Only Look Once: Unified, Real-Time Object Detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf) (**CVPR** 2016)

## 2D Segmentation

### Algorithm

* [Mask R-CNN](https://arxiv.org/abs/1703.06870) (**ICCV** 2017)

## 3D Segmentation

### Algorithm

* [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) (**openaccess.thecvf** 2017)

* [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413) (**NIPS** 2017)

* [PointCNN](https://arxiv.org/abs/1801.07791) (**arxiv** 2018)

# Reinforcement Learning

## Environment

* [MuJoCo: A physics engine for model-based control](http://ieeexplore.ieee.org/abstract/document/6386109/?reload=true) (**International Conference on Intelligent Robots and Systems** 2012)

* [OpenAI Gym](https://arxiv.org/abs/1606.01540) (**arxiv** 2016)

## Algorithm
* 【DQN】[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (**NIPS workshop** 2013)

* 【DPG】[Deterministic Policy Gradient Algorithms](https://hal.inria.fr/hal-00938992/) (**ICML** 2014)

* 【TRPO】[Trust Region Policy Optimization](http://proceedings.mlr.press/v37/schulman15.pdf) (**ICML** 2015)

* 【A3C】[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) (**ICML** 2016)

* 【DDPG】[Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) (**ICLR** 2016)

* 【NAF】[Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748) (**arxiv** 2016)

* 【ACER】[Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224) (**arxiv** 2016)

* 【GAIL】[Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476) (**NIPS** 2016)

* [Q-PROP: SAMPLE-EFFICIENT POLICY GRADIENT WITH AN OFF-POLICY CRITIC](https://arxiv.org/abs/1611.02247) (**ICLR** 2017)

* 【PPO】[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (**arxiv** 2017)

* [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286) (**arxiv** 2017)

* 【ACKTR】[Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/abs/1708.05144) (**NIPS** 2017)

* 【HER】[Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (**NPIS** 2017)

# Robot

## Grasping

### Grasping Unknown Objects

* [Ranking the good points: A comprehensive method for humanoid robots to grasp unknown objects](http://poeticonpp.csri-web.org:8989/PoeticonPlus/publications/1342_Gori_etal2013.pdf) (**International Conference on Advanced Robotics** 2013)

* [Model-Free Segmentation and Grasp Selection of Unknown Stacked Objects](http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/ECCV_2014/papers/8693/86930659.pdf) (**European Conference on Computer Vision** 2014)

### Grasping in Cluttered Environment

* [Robotic Pick-and-Place of Novel Objects in Clutter with Multi-Affordance Grasping and Cross-Domain Image Matching](http://vision.princeton.edu/projects/2017/arc/paper.pdf) (**arxiv** 2017)

### Grasping via Segmentation

* [Grasping novel objects with depth segmentation](http://www.robotics.stanford.edu/~ang/papers/iros10-GraspingWithDepthSegmentation.pdf) (**IROS** 2010)

* [3D scene segmentation for autonomous robot grasping](https://www.researchgate.net/publication/261353757_3D_scene_segmentation_for_autonomous_robot_grasping) (**IROS** 2012)

### Grasping Points Selection
* [GP-GPIS-OPT: Grasp planning with shape uncertainty using Gaussian process implicit surfaces and Sequential Convex Programming](http://rll.berkeley.edu/~sachin/papers/Mahler-ICRA2015.pdf) (**ICRA** 2015)

* [Dex-Net 1.0: A Cloud-Based Network of 3D Objects for Robust Grasp Planning Using a Multi-Armed Bandit Model with Correlated Rewards](http://goldberg.berkeley.edu/pubs/icra16-submitted-Dex-Net.pdf) (**ICRA** 2016)

* [Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics](https://arxiv.org/abs/1703.09312) (**arxiv** 2017)

## Active Perception
