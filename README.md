# SimplifiedFmapsLearning

This is the official repository of the paper [Memory Scalable and Simplified Functional Map Learning](https://openaccess.thecvf.com/content/CVPR2024/html/Magnet_Memory-Scalable_and_Simplified_Functional_Map_Learning_CVPR_2024_paper.html), published at CVPR 2024

This is a v0.1 of the codebase which should be able to reproduce the results exactly. There is no explicit dependency on [ScalableDenseMaps](https://github.com/RobinMagnet/ScalableDenseMaps) in this version, an early versio of the package is provided in the DiffZo folder.

# Project Overview

A **cleaner** and **simplified** version of this codebase is in preparation and will soon be released. It will integrate more directly with [ScalableDenseMaps](https://github.com/RobinMagnet/ScalableDenseMaps).

This implementation is largely based on [AttentiveFMaps](https://github.com/craigleili/AttentiveFMaps), with several modifications:
- A new training pipeline
- Enhancements related to Differentiable ZoomOut

## Training
To train the model, use
```bash
python trainer_dzo.py run_mode=train run_cfg=path/to/cfg/config.yml
```

The data can be found on the [AttentiveFMaps](https://github.com/craigleili/AttentiveFMaps) github.

## Testing and Pretrained Models

Original pretrained checkpoints are available [here](https://www.lix.polytechnique.fr/Labo/Robin.Magnet/CVPR24/res_cache.zip), although some files appear to be corrupted.
An updated set of checkpoints, compatible with PyTorch 2.4.1 and CUDA 12.1, is available [on Google Drive](https://drive.google.com/file/d/1ebo5iGGavkaEat40QfdS4N-MInN-c3RQ/view?usp=sharing).

**Note**: Results can vary significantly depending on the Torch and CUDA versions used.

I am currently in the process of reconstructing the original training environment and will update the checkpoints accordingly. A Dockerfile for the environment used with the second set of checkpoints is also available upon request.

You can then run evaluation using this code:

```bash
python trainer_dzo.py run_mode=test run_ckpt=path/to/ckpt/ckpt.pth
```


You can run your own evaluation using the pretrained checkpoints, where the *DiffusionNet*'s weights are stored in the `feature_extractor` key.

Let me know if you'd like help with the introduction or usage instructions!

 # Citing this work

 If you use this work, please cite

 ```bibtex
@inproceedings{magnetMemoryScalable2024,
  title = {Memory Scalable and Simplified Functional Map Learning},
  booktitle = {2024 {{IEEE}}/{{CVF Conference}} on {{Computer Vision}} and {{Pattern Recognition}} ({{CVPR}})},
  author = {Magnet, Robin and Ovsjanikov, Maks},
  year = {2024},
  publisher = {IEEE},
}
```
