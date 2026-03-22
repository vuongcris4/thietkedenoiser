import wandb
api = wandb.Api()
run = api.run("vuongcris4-hcmute/thietkedenoiser/jwqcmxim")
print(run.history())
