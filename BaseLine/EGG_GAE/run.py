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
from utils import config_preprocess, get_dataloader
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
    ARMSE_list, AMAE_list, acc_list = [], [], []
    all_ARMSE, all_AMAE, all_acc = 0, 0, 0
    for i in range(1):
        setup_cuda(cfg)
        print(OmegaConf.to_yaml(cfg))

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
        mean, scale = data_pipeline(dataset_name=dataset_name,
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

        value_cat =  [0, 1, 2, 3, 5, 6]
        # continuous_cols = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        # continuous_cols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56]
        continuous_cols = [4, 7,8]
        num_cur_index = 0
        val_x = datamodule.val_dataloader().dataset.features_clean
        for col_idx in range(val_x.shape[1]):
            if col_idx in continuous_cols:
                val_x[:, col_idx] = val_x[: ,col_idx] * scale[num_cur_index] + mean[num_cur_index]
                num_cur_index += 1
        val_label = datamodule.val_dataloader().dataset.labels.reshape(-1,1)
        val_data = pd.DataFrame(np.concatenate((val_x, val_label), axis=1))


        trainer.fit(model, datamodule)

        test_dataloader = datamodule.train_dataloader().dataset
        a = test_dataloader.features_clean
        a_ditry = test_dataloader.features
        a_m = test_dataloader.MASK_init

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
        # ori_data = []
        label_data = []
        M_data = []
        num_data = []
        cat_data = []
        model.model.to('cuda:0')
        # logits, y, num_rec, cat_outputs = model.model.predict(miss_data,label.squeeze(-1).to(torch.long))
        # data_m = M.detach().cpu().numpy()
        # num_rec = num_rec.detach().cpu().numpy()
        # ori_data = ori_data.cpu().numpy()
        # impute_data = num_rec * (1 - data_m) + ori_data * data_m
        # ori_data = ori_data * scale + mean
        # impute_data = impute_data * scale + mean
        for batch in test_dataloader_new:
            batch_M, batch_ori_data, batch_miss_data, batch_label = batch
            logits, y, num_rec, cat_outputs = model.model.predict(batch_miss_data.to(torch.bfloat16), batch_label.squeeze(-1).to(torch.long))
            # 将num_rec和cat_outputs拼接
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
            if col in continuous_cols:
                impute_data[:, col] = impute_num_numpy[:, num_index].copy()
                num_index += 1
            else:
                impute_data[:, col] = impute_cat_numpy[:, cat_index].copy()
                cat_index += 1

        # impute_cat_numpy = torch.cat(cat_data, dim=0).detach().numpy()
        M_numpy = torch.cat(M_data, dim=0).cpu().detach().numpy()
        ori_numpy = torch.cat(res_ori_data, dim=0).numpy()
        num_cur_index = 0
        for col_idx in range(ori_numpy.shape[1]):
            if col_idx in continuous_cols:
                ori_numpy[:, col_idx] = ori_numpy[: ,col_idx] * scale[num_cur_index] + mean[num_cur_index]
                num_cur_index += 1

        impute_num_numpy = M_numpy * ori_numpy + (1 - M_numpy) * impute_data
        # ARMSE,AMAE = get_RMSE(ori_data,impute_data)

        values = [x for x in range(ori_numpy.shape[1])]

        ori_pd = pd.DataFrame(ori_numpy, columns=values)
        copy_ori_data = ori_pd.copy()
        device = "cuda:0"

        ori_code, enc = categorical_to_code(copy_ori_data, value_cat, enc=None)
        fill_data_mean = pd.DataFrame(impute_num_numpy, columns=values)

        # cat_to_code_data, enc = categorical_to_code(ori_pd.copy(), value_cat, enc)
        # cat_to_code_data.columns = [x for x in range(cat_to_code_data.shape[1])]
        # fields, feed_data = Data_convert(cat_to_code_data, "minmax", continuous_cols)
        # impute_data_code = torch.tensor(feed_data.values, dtype=torch.float).to(device)
        acc = get_down_acc(fill_data_mean, label_data, val_data, 2, device, value_cat, continuous_cols, enc=None)
        RMSE, MAE = errorLoss(fill_data_mean, ori_pd, M_numpy, value_cat, continuous_cols, enc)
        print("ARMSE为：{:.4f}，AMAE为：{:.4f}，Acc为：{:.4f}".format(RMSE, MAE, acc))
        # ARMSE_list.append(RMSE)
        # AMAE_list.append(MAE)
        # acc_list.append(acc)
        # all_ARMSE = all_ARMSE + RMSE
        # all_AMAE = all_AMAE + MAE
        # all_acc = all_acc + acc
    # ARMSE = all_ARMSE / 1.
    # AMAE = all_AMAE / 1.
    # Acc = all_acc / 1.
    # var_ARMSE = np.std(ARMSE_list)
    # var_AMAE = np.std(AMAE_list)
    # var_acc = np.std(acc_list)
    # print("ARMSE为：{:.4f}+var:{:.4f}，AMAE为：{:.4f}+var:{:.4f}，Acc为：{:.4f}+var:{:.4f}".format(ARMSE, var_ARMSE, AMAE,
    #                                                                                           var_AMAE, Acc, var_acc))



# def cat_data(num_numpy, cat_numpy, cat_idx):

    # val_data = T_dataloader.features
    # val_label = T_dataloader.labels
    #
    # train_data = res_data_numpy * scale + mean
    # val_data = val_data * scale + mean
    # Acc = get_down_acc(train_data,label_data,val_data,val_label)
    # print(Acc)
    # Train
    # #
    # trainer.test(model=model, dataloaders=datamodule.test_dataloader(), ckpt_path="best")

    # print('Training is done!')

    # return mean, scale


#
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()
        # print(torch.cuda.device_count())
    # cfg = DictConfig()
    # data = np.load('noised_datasets/no_exp/wine/wine.npz')
    # ori_data = data['X_train_clean']
    # test_data = data['X_train_deg_freq']
    # mask = data['mask_init_train']
    # mean = data['mean']
    # scale = data['scale']
        main()

    # ARMSE,AMAE = get_RMSE(ori_data, test_data, mask, mean, scale)
    # print(ARMSE,AMAE)

    # array3 = data['X_train_deg_freq']

