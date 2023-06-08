Run docker:

```
docker run -it -p 9999:9999 --rm --memory=32g --shm-size=32g --cpuset-cpus=0-2 --gpus '"device=1"' --security-opt label=disable --name er --init  -v /home/eromanenkova:/home -v /home/eromanenkova/stash:/home/stash eromanenkova/cpd_video
```

Run script:

Train
```
python3 script.py --model bce --block-type tcl3d --bias-rank 1 --experiments-name road_accidents --epochs 200
```

Test (`1665087354` is timestamp of model in command; for new trained models it will be smth like 221010T112811)
```
python3 script_test.py 221128T144649 --model bce  --experiments-name road_accidents -tn 25
```

Results can be found in `saves/results/road_accidents/metrics`.

For testing, in `utils/kl_cpd.py`, line 330 `pred_out = torch.tanh(pred_out * 10 ** 7)`, scale for `explosion` wa in `[1e4, 1e5, 1e6, 1e7]`.
For training, `--bias-rank` can be `[4, 8, -1]`. Also we can try `--emb-dim "64,8,8"` and `--hid-dim "32,4,4"` (see Table 2 in report).



Make configs based on csv with selected configurations and run all configs in folder

```
python3 make_config_folder.py explosion configs/config_bce_v5_template_seed.yaml configs_grid saves/results/best_explosions_part.csv

python3 run_batch.py configs_grid
```