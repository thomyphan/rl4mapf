# Reinforcement Learning Techniques for Multi-Agent Path Finding

## Featured Algorithm

- CACTUS: Confidence-based Auto- Curriculum for Team Update Stability [1]

## Prerequisites
Run these commands
```
cd instances
mkdir primal_test_envs
```
### Training maps
Are generated for each training run in `run_training.py` using `cactus.env.env_generator`.

### Test maps
Go to the Google Drive referenced by the PRIMAL Github repository. Download the archive with all PRIMAL test maps [2] and unpack it in `instances/primal_test_envs`

## Running the Code

### Training

Run training of all MARL algorithms in the paper with (creates a folder mit `results.json` and `actor.pth` for evaluation):

```
python run_training.py
```

The command will create a folder `output/` with named result folders per MARL algorithm.

### Test

Run evaluation with (parameter `filename` specifies the result folder with `actor.pth`)
```
python eval.py <filename> <map_size> <density>
```

The completion rates are printed on the command line and can be redirected into a text or JSON file for post-processing.

## References

[1] T. Phan et al., *"Confidence-Based Curriculum Learning for Multi-Agent Path Finding"*, AAMAS 2024 (To appear)

[2] G. Sartoretti et al., *"PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning"*, RA-L 2019
