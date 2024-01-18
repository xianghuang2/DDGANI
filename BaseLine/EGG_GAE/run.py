import os
import warnings
# hydra
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import torch
# pytorch-lightning related imports
from pytorch_lightning import Trainer
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import TensorDataset, DataLoader

from method import LitModel
from inference.val_logic import ValCallback
from inference.test_logic import TestCallback

from data_prep.pipeline import data_pipeline
from util import categorical_to_code, Data_convert, get_down_acc, errorLoss
from BaseLine.EGG_GAE.utils import config_preprocess, get_dataloader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

def setup_cuda(cfg: DictConfig):

    print("DEVICE COUNT: ",torch.cuda.device_count())
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.trainer.cuda_number
    
    print("DEVICE COUNT: ",torch.cuda.device_count())

@hydra.main(config_path='./configs', config_name='defaults')
def main(cfg: DictConfig):
    setup_cuda(cfg)
    # print(OmegaConf.to_yaml(cfg))
    dataloader_cfg = cfg.dataloader
    dataset_name = dataloader_cfg.dataset_name
    experiment_name = dataloader_cfg.experiment_name
    val_size = dataloader_cfg.val_size
    data_split_seed = dataloader_cfg.data_seed
    miss_algo = dataloader_cfg.miss_algo
    def_fill_val = dataloader_cfg.imputation.fill_val
    p_miss = dataloader_cfg.imputation.init_corrupt
    miss_seed = dataloader_cfg.imputation.seed
    scale_type = dataloader_cfg.imputation.scale_type

    # Prepare data
    mean, scale, cat_cols, con_cols = data_pipeline(dataset_name=dataset_name,
                  experiment_name=experiment_name,
                  def_fill_val=def_fill_val,
                  p_miss=p_miss, # initial data corruption
                  miss_algo=miss_algo,
                  miss_seed=miss_seed,
                  val_size=val_size,
                  data_split_seed=data_split_seed,
                  scale_type=scale_type
                  )

    cfg.wadb.logger_name = f'{cfg.dataloader.dataset_name}_{cfg.model.edge_generation_type}_k(noSKIPCON)'

    # # # Configure weight and biases
    if cfg.trainer.is_logger_enabled == True:
        logger = pl_loggers.WandbLogger(
            project=cfg.wadb.logger_project_name,
            name=cfg.wadb.logger_name if cfg.wadb.logger_name != 'None' else None,
            entity=cfg.wadb.entity,

            )

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        monitor='val_ens_aucroc',
        mode='max',
        save_top_k=1,
        save_last=False,
        verbose=True,
        dirpath="checkpoints",
        filename="epoch_{epoch:03d}",
    )
    early_stopping = EarlyStopping(monitor="val_ens_aucroc", mode="max", min_delta=0.00, patience=3)

    # Setup dataloader and model

    datamodule = get_dataloader(cfg)

    # Get additional statistics from dataset
    cfg = config_preprocess(cfg, datamodule)

    callbacks = [ValCallback(num_classes=cfg.model.outsize, device='cuda:0'),
                 TestCallback(num_classes=cfg.model.outsize, device='cuda:0'),
                 LearningRateMonitor("step"), checkpoint_callback, early_stopping]

    # Configure trained
    trainer = Trainer(devices=1, accelerator='gpu',
        logger=logger if cfg.trainer.is_logger_enabled else False,
        num_sanity_val_steps=-1,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        max_epochs=cfg.model.opt.max_epochs,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=callbacks if cfg.trainer.is_logger_enabled else [])
    # a = datamodule.train_dataloader().dataset
    torch.set_float32_matmul_precision('medium')
    model = LitModel(datamodule=datamodule, cfg=cfg).cuda()
    # checkpoint = torch.load('outputs/2023-09-01/17-09-37/checkpoints/epoch=8-step=18.ckpt')
    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()

    num_cur_index = 0
    val_x = datamodule.test_dataloader().dataset.features_clean
    for col_idx in range(val_x.shape[1]):
        if col_idx in con_cols:
            val_x[:, col_idx] = val_x[: ,col_idx] * scale[num_cur_index] + mean[num_cur_index]
            num_cur_index += 1
    val_label = datamodule.test_dataloader().dataset.labels.reshape(-1,1)
    val_data = pd.DataFrame(np.concatenate((val_x, val_label), axis=1))



    test_dataloader = datamodule.train_dataloader().dataset
    a = test_dataloader.features_clean
    a_ditry = test_dataloader.features
    a_m = test_dataloader.MASK_init
    trainer.fit(model, datamodule)

    M = test_dataloader.MASK_init.squeeze(-1).to('cuda:0')
    miss_data = torch.from_numpy(test_dataloader.features).to('cuda:0')
    ori_data = torch.from_numpy(test_dataloader.features_clean)
    # clean_data = M * clean_data
    label = torch.from_numpy(test_dataloader.labels).unsqueeze(-1).to('cuda:0')
    combined_dataset = TensorDataset(M, ori_data, miss_data, label)
    bs = miss_data.shape[0]
    test_dataloader_new = DataLoader(combined_dataset, batch_size=100, shuffle=True)
    res_data = []
    res_ori_data = []
    label_data = []
    M_data = []
    num_data = []
    cat_data = []
    model.model.to('cuda:0')
    for batch in test_dataloader_new:
        batch_M, batch_ori_data, batch_miss_data, batch_label = batch
        logits, y, num_rec, cat_outputs = model.model.predict(batch_miss_data.to(torch.bfloat16), batch_label.squeeze(-1).to(torch.long))
        if len(cat_outputs) > 0:
            labels = [torch.argmax(output, dim=1, keepdim=True) for output in cat_outputs]
            final_output = torch.cat(labels, dim=1)
            cat_data.append(final_output)
        num_data.append(num_rec)
        label_data.append(batch_label)
        M_data.append(batch_M)
        res_ori_data.append(batch_ori_data)
    label_data = pd.DataFrame(torch.cat(label_data, dim=0).cpu().detach().numpy())
    impute_num_numpy = torch.cat(num_data, dim=0).cpu().detach().numpy()
    impute_num_numpy = impute_num_numpy * scale + mean

    if len(cat_outputs) > 0:
        impute_cat_numpy = torch.cat(cat_data, dim=0).cpu().detach().numpy()
        col_num = impute_cat_numpy.shape[1]+impute_num_numpy.shape[1]
        impute_data = np.zeros((impute_num_numpy.shape[0], col_num))
    else:
        impute_data = np.zeros((impute_num_numpy.shape[0], impute_num_numpy.shape[1]))

    num_index = 0
    cat_index = 0
    for col in range(impute_data.shape[1]):
        if col in con_cols:
            impute_data[:, col] = impute_num_numpy[:, num_index].copy()
            num_index += 1
        else:
            impute_data[:, col] = impute_cat_numpy[:, cat_index].copy()
            cat_index += 1

    M_numpy = torch.cat(M_data, dim=0).cpu().detach().numpy()
    ori_numpy = torch.cat(res_ori_data, dim=0).numpy()
    num_cur_index = 0
    for col_idx in range(ori_numpy.shape[1]):
        if col_idx in con_cols:
            ori_numpy[:, col_idx] = ori_numpy[: ,col_idx] * scale[num_cur_index] + mean[num_cur_index]
            num_cur_index += 1

    impute_num_numpy = M_numpy * ori_numpy + (1 - M_numpy) * impute_data
    # ARMSE,AMAE = get_RMSE(ori_data,impute_data)

    values = [x for x in range(ori_numpy.shape[1])]

    ori_pd = pd.DataFrame(ori_numpy, columns=values)
    copy_ori_data = ori_pd.copy()
    device = "cuda:0"

    ori_code, enc = categorical_to_code(copy_ori_data, cat_cols, enc=None)
    fill_data_mean = pd.DataFrame(impute_num_numpy, columns=values)

    value_cat = cat_cols

    acc = get_down_acc(fill_data_mean, label_data, val_data, value_cat, con_cols, enc=None, seed=42)
    RMSE, MAE = errorLoss(fill_data_mean, ori_pd, M_numpy, value_cat, con_cols, enc)
    print("ARMSE为：{:.4f}，AMAE为：{:.4f}，Acc为：{:.4f}".format(RMSE, MAE, acc))



if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()
        main()

