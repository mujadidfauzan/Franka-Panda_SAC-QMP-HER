# Panda SAC QMP-HER

## QMP-HER training

The target policy uses a goal-agnostic 39-value state plus Gymnasium's
`achieved_goal` and `desired_goal`. This keeps HER relabeling consistent while
the grasp and insert projectors still produce the observation layouts expected
by their pretrained primitive policies.

Start a fresh QMP-HER run after this observation-schema change:

```bash
python -m scripts.train_qmpher_sac --total-timesteps 5000000
```

Check the primitive composition baseline before admitting the target actor:

```bash
python -m scripts.train_qmpher_sac \
  --total-timesteps 1000000 \
  --target-max-admission-probability 0 \
  --run-name panda_qmpher_primitive_baseline
```

The default selector now latches the insert stage after a verified grasp,
locks the selected primitive for the complete stage, disables full-random QMP
actions at 500k steps, and gradually admits the deterministic target actor up
to 20 percent of new stages. TensorBoard records the stage, stage lock, target
admission, grasp verification, and object-loss diagnostics under `qmp/`.

Checkpoints created with the old 45-value target state and their replay buffers
are not compatible with the new target policy. The pretrained grasp and insert
primitive checkpoints remain compatible.
