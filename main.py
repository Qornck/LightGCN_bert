import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
import os

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
bert_file = utils.getFileName_bert()
mlp_file = utils.getFileName_MLP()

print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        Recmodel.bert.load_state_dict(torch.load(bert_file, map_location=torch.device('cpu')))
        Recmodel.mlp.load_state_dict(torch.load(mlp_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    print("tensorboard on")
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

best_results = {'precision': 0,
               'recall': 0,
               'ndcg': 0}
try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            cprint("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            if results['recall'][0] > best_results['recall']:
                best_results['recall'] = results['recall'][0]
                best_results['precision'] = results['precision'][0]
                best_results['ndcg'] = results['ndcg'][0]
            torch.save(Recmodel.state_dict(), weight_file)
            torch.save(Recmodel.bert.state_dict(), bert_file)
            torch.save(Recmodel.mlp.state_dict(), mlp_file)
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
finally:
    if world.tensorboard:
        w.close()

cprint(best_results)

run_folder = os.listdir(world.BOARD_PATH)[-1]
with open(f"{world.BOARD_PATH}/{run_folder}/results.txt", 'w') as f:
    f.write(str(best_results))