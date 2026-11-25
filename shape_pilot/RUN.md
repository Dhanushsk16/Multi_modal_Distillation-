# 10M student shape pilot (lab server)

Code lives in ~/code/shape_pilot. Everything written at runtime goes to
/mnt/wwn-0x5000c500e64c5cc0/dhanushsk/BTPData (see scripts/env.sh).

One time (the server has no internet; everything comes from offline.tar):
    tar xf $DATA/cache/offline.tar -C $DATA/cache       # -> $DATA/cache/offline/
    bash ~/code/shape_pilot/scripts/setup_env.sh        # Miniconda + env "peptide"
    source ~/.bashrc

Each run (at night):
    nvidia-smi; top                                    # check nobody else is on it
    conda activate peptide
    tmux new -s pilot
    bash ~/code/shape_pilot/scripts/run_shape.sh s256x15
    # detach Ctrl-b d | reattach: tmux attach -t pilot

The script refuses to start if another process is on the GPU, stops itself at
06:45, and resumes from its last checkpoint if re-run.

Outputs:
    logs      $DATA/logs/shape_pilot/<shape>.log
    weights   $DATA/checkpoints/students/, $DATA/checkpoints/shape_pilot/<shape>/
    metrics   $DATA/results/shape_pilot/summary.csv, latency/
