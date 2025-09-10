gitx# Running ML4FIR Training Experiments in `experiments/configs/simple_ml/`

This guide explains how to run all ML4FIR training experiments in the `experiments/configs/simple_ml/` folder and commit the results to your repository.

## Steps for Each Experiment

1. **Run the training command:**

   ```bash
   
   ```
   Replace `<experiment_file>.json` with the name of the experiment config file you want to run (e.g., `exp_il_10.json`).

2. **Add results and artifacts to git:**

   ```bash
git p
   git commit -m "upload <experiment_file> experience"
   git push
   ```
   Replace `<experiment_file>` with the name of the experiment you just ran.

## Example for All Experiments

To run all experiments in the folder, repeat the steps above for each file:

```bash
for exp in experiments/configs/simple_ml/*.json; do
    ml4fir train "$exp"
    git add mlruns/*
    git add mlartifacts/*
    git add experiments/*
    git add results/*
    git commit -m "upload $(basename "$exp" .json) experience"
    git push
done
```

## Notes
- Make sure you have the required environment and permissions to run `ml4fir` and git commands.
- This will commit and push all experiment results and artifacts to your repository after each run.
- You can customize the commit message as needed.
