## phase 3 experiments (increase in steps does fitness increase? )
#uv run python main.py --n-states 80 --n-clusters 8 --n-observers 40 --n-steps 200000 --seeds 1 2 3 --prefix exp_01_gen_20_steps_200000
uv run python main.py --n-states 80 --n-clusters 8 --n-observers 40 --selection-interval 1000 --n-steps 200000 --seeds 1 2 3 --prefix exp_02_gen_50_steps_200000
uv run python main.py --n-states 80 --n-clusters 8 --n-observers 40 --selection-interval 2000 --n-steps 200000 --seeds 1 2 3 --prefix exp_03_gen_1000_steps_200000

## phase 4 experiments (increase what happens if we increase observers? )

uv run python main.py --n-states 80 --n-clusters 8 --n-observers 80 --n-steps 100000 --seeds 1 2 3 --prefix exp_01_gen_20_steps_100000_ob80
uv run python main.py --n-states 80 --n-clusters 8 --n-observers 80 --n-steps 200000 --seeds 1 2 3 --prefix exp_02_gen_20_steps_200000_ob80
uv run python main.py --n-states 80 --n-clusters 8 --n-observers 180 --n-steps 400000 --seeds 1 2 3 --prefix exp_02_gen_20_steps_200000_ob80

