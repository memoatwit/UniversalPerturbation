# Universal Perturbation

## Multi-View Adversarial Images with a Universal Perturbation

This paper proposes a novel universal perturbation method that, despite operating on 2D images, achieves a 3D-like attack capability by generating adversarial examples robust to variations in viewing angle, offering a more practical and scalable alternative to 3D adversarial attacks. This generalizable approach provides multi-view adversarial attacks while keeping the computational efficiency and practical aspects of single-view methods.

The paper is accepted to ICAIA and a preprint is submitted to arXiv. 

## Installation
Pre-reqs: 
- Blender Version 4.0.2 (4.0.2 2023-12-05)
- Python 3.10 
A sample environment  is included in our requirements and is created with:
```
conda create -n up python=3.10
conda activate up
pip install tensorflow numpy pillow torch pandas matplotlib cleverhans bpy
pip freeze > requirements.txt
```

You can test the installation with the included sample image by running:
> python /Users/memo/Downloads/UniversalPerturbation-testing/main.py

This will generate and compare the proposed Universal attack to our and CleverHans' FGSM implementation.

> iteration: 0, loss: [0.39462262] <br>
> (1,) (224, 224, 3) <br>
> Universal: [[('n03903868', 'pedestal', 0.24639696), ('n03482405', 'hamper', 0.14491189), ('n03127925', 'crate', 0.120108284), ('n03998194', 'prayer_rug', 0.03229109), ('n02971356', 'carton', 0.031419415)]] <br>
> FGSM: [[('n03903868', 'pedestal', 0.13343877), ('n04141975', 'scale', 0.106996596), ('n03482405', 'hamper', 0.09244215), ('n03127925', 'crate', 0.05132524), ('n04553703', 'washbasin', 0.015889827)]] <br>
> Cleverhans FGSM: [[('n03903868', 'pedestal', 0.1338335), ('n04141975', 'scale', 0.10725212), ('n03482405', 'hamper', 0.091944754), ('n03127925', 'crate', 0.05115742), ('n04553703', 'washbasin', 0.015928818)]] <br>


### To Do
[ ] Add Clip attack to torch <br>
[ ] Add plot generation class <br>
[ ] Add targeted attacks <br>
