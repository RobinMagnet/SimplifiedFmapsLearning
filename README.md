# SimplifiedFmapsLearning

This is a v0.1 of the codebase which should be able to reproduce the results exactly. There is no explicit dependency on [ScalableDenseMaps](https://github.com/RobinMagnet/ScalableDenseMaps) in this version, an early versio of the package is provided in the DiffZo folder.

A **simpler** and **cleaner** version of the codebase will be released soon, relying cleanly on [ScalableDenseMaps](https://github.com/RobinMagnet/ScalableDenseMaps).

The code is essentially the one from [AttentiveFMaps](https://github.com/craigleili/AttentiveFMaps), but providing a new trainer, and adding things related to Differentiable ZoomOut.

## Training
To train the model, use
```bash
python trainer_dzo.py run_mode=train run_cfg=path/to/cfg/config.yml
```

The data can be found on the [AttentiveFMaps](https://github.com/craigleili/AttentiveFMaps) github.

## Testing

You can download pretrained models at [this address](https://www.lix.polytechnique.fr/Labo/Robin.Magnet/CVPR24/res_cache.zip). You can then run evaluation using this code:

```bash
python trainer_dzo.py run_mode=test run_ckpt=path/to/ckpt/ckpt.pth
```


You can run your own evaluation using the pretrained checkpoints, where the *DiffusionNet*'s weights are stored in the `feature_extractor` key.


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