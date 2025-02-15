<div align="center">
    <h1>One Noise to Rule Them All: Multi-View Adversarial Attacks with Universal Perturbation</h1>

  <a href='https://arxiv.org/abs/2404.02287'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a>
    <div>
        <a href='https://github.com/memoatwit' target='_blank'>M. Ergezer <sup>1, 2</sup></a>&emsp;
        <a href ='https://github.com/phatgd' target='_blank'>P. Duong <sup>1</sup> </a>&emsp;
    <a href='https://github.com/greenchris10' target='_blank'>C. Green <sup>1</sup></a>&emsp;
      <a href='https://github.com/Tommy-Nguyen-cpu' target='_blank'>T. Nguyen <sup>1</sup> </a>&emsp;
          <a href='https://github.com/azeybey' target='_blank'> A. Zeybey <sup>1</sup></a>&emsp;
  </div>
  <br>
  <div>
      <sup>1</sup> Wentworth Institute of Technology <br>
      <sup>2</sup> Dr. Ergezer holds concurrent appointments as an Associate Professor at Wentworth Institute of Technology and as an Amazon Visiting Academic. This paper describes work performed at Wentworth Institute of Technology and is not associated with Amazon.
  </div>
</div>

## Abstract
This paper presents a novel universal perturbation method for generating robust multi-view adversarial examples in 3D object recognition. Unlike conventional attacks limited to single views, our approach operates on multiple 2D images, offering a practical and scalable solution for enhancing model scalability and robustness. This generalizable method bridges the gap between 2D perturbations and 3D-like attack capabilities, making it suitable for real-world applications.

Existing adversarial attacks may become ineffective when images undergo transformations like changes in lighting, camera position, or natural deformations. We address this challenge by crafting a single universal noise perturbation applicable to various object views. Experiments on diverse rendered 3D objects demonstrate the effectiveness of our approach. The universal perturbation successfully identified a single adversarial noise for each given set of 3D object renders from multiple poses and viewpoints. Compared to single-view attacks, our universal attacks lower classification confidence across multiple viewing angles, especially at low noise levels. A sample implementation is made available [here](https://github.com/memoatwit/UniversalPerturbation).


## Article
The paper is presented at the 2nd International Conference on Artificial Intelligence and Applications (ICAIA 2024) and a preprint is submitted to [arXiv](https://arxiv.org/abs/2404.02287) and can be cited as:
```
@misc{ergezer2024noiseruleallmultiview,
      title={One Noise to Rule Them All: Multi-View Adversarial Attacks with Universal Perturbation}, 
      author={Mehmet Ergezer and Phat Duong and Christian Green and Tommy Nguyen and Abdurrahman Zeybey},
      year={2024},
      eprint={2404.02287},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.02287}, 
}
```

## Installation
Pre-reqs: 
- Blender Version 4.0.2 (4.0.2 2023-12-05)
- Python 3.10 (required version for Blender API compatibility) <br>
A sample environment  is included in our requirements and is created with:
```
conda create -n up python=3.10
conda activate up
pip install -r requirements.txt
```

You can test the installation with the included sample image by running the following: <br>
```python main.py```

This will generate and compare the proposed Universal attack to [CleverHans'](https://github.com/cleverhans-lab/cleverhans) FGSM implementation.
```
> Target label: `desk` 
> iteration: 0, loss: [0.39462262] 
> (1,) (224, 224, 3) 
> Universal: [[('n03903868', 'pedestal', 0.24639696), ('n03482405', 'hamper', 0.14491189), ('n03127925', 'crate', 0.120108284), ('n03998194', 'prayer_rug', 0.03229109), ('n02971356', 'carton', 0.031419415)]] 
> Our FGSM: [[('n03903868', 'pedestal', 0.13343877), ('n04141975', 'scale', 0.106996596), ('n03482405', 'hamper', 0.09244215), ('n03127925', 'crate', 0.05132524), ('n04553703', 'washbasin', 0.015889827)]] 
> Cleverhans FGSM: [[('n03903868', 'pedestal', 0.1338335), ('n04141975', 'scale', 0.10725212), ('n03482405', 'hamper', 0.091944754), ('n03127925', 'crate', 0.05115742), ('n04553703', 'washbasin', 0.015928818)]] 
```

## Related Work
- A. Zeybey, M. Ergezer, and T. Nguyen. ``Gaussian Splatting Under Attack: Investigating Adversarial Noise in 3D Objects.'' NeurIPS Safe Generative AI Workshop, 2024 [ArXiv](https://arxiv.org/abs/2412.02803).
-  C. Green, M. Ergezer,  A. Zeybey. ``Targeted View-Invariant Adversarial Perturbations for 3D Object Recognition''. AAAI-25 Workshop on Artificial Intelligence for Cyber Security (AICS), 2025 [ArXiv](https://arxiv.org/abs/2412.13376).
-  T. Nguyen, M. Ergezer, C. Green (2025). ``AdvIRL: Reinforcement Learning-Based Adversarial Attacks on 3D NeRF Models''. AAAI-25 Workshop on Artificial Intelligence for Cyber Security (AICS), 2025 [ArXiv](https://arxiv.org/abs/2412.16213]) and [Project Page](https://tommy-nguyen-cpu.github.io/AdvIRL/)
