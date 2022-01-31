import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from data import tflog2pandas as parsetf
import shutil
import yaml
from factory import *
from file_utils import *
from experiment import Experiment
from dataloader import *
from graphs import *


def plot_train_val_loss(df: pd.DataFrame, savedir: str) -> None:
    df_train_loss = df[df.metric == "train_loss_epoch"]
    df_train_loss.reset_index(inplace=True)
    df_val_loss = df[df["metric"] == "val_loss"]
    df_val_loss.reset_index(inplace=True)
    df_res = pd.DataFrame(index=df_train_loss.index)
    df_res["train_loss"] = df_train_loss["value"]
    df_res["val_loss"] = df_val_loss["value"]

    if not df_res.empty:
        df_res.plot(figsize=(8, 6))
        plt.xlabel("Epochs")
        plt.ylabel("Combined Loss")
        plt.savefig(os.path.join(savedir, "train_val_loss.png"), bbox_inches="tight")
        plt.close()


def plot_val_acc_mIoU(df: pd.DataFrame, savedir: str) -> None:
    df_val_mIoU = df[df["metric"] == "val_mIoU"]
    df_val_mIoU.reset_index(inplace=True)
    df_val_acc = df[df["metric"] == "val_acc"]
    df_val_acc.reset_index(inplace=True)
    df_res = pd.DataFrame(index=df_val_acc.index)
    df_res["val_acc"] = df_val_acc["value"]
    df_res["val_mIoU"] = df_val_mIoU["value"]

    if not df_res.empty:
        df_res.plot(figsize=(8, 6))
        plt.xlabel("Epochs")
        plt.ylabel("")
        plt.savefig(os.path.join(savedir, "acc_mIoU.png"), bbox_inches="tight")
        plt.close()


def plot_val_rms(df: pd.DataFrame, savedir: str) -> None:
    df_res = df[df["metric"] == "val_rms"]
    df_res.reset_index(inplace=True)

    if not df_res.empty:
        df_res["value"].plot(figsize=(8, 6))
        plt.xlabel("Epochs")
        plt.ylabel("Root Mean Squared Error")
        plt.savefig(os.path.join(savedir, "val_rms_err.png"), bbox_inches="tight")
        plt.close()


def map_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps the epoch to steps.
    """
    epoch = df[df.metric == "epoch"][["value", "step"]].rename(
        columns={"value": "epoch", "step": "id"}
    )
    return df.merge(epoch, left_on="step", right_on="id")[
        ["metric", "value", "epoch", "step"]
    ]


def get_metrics(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """
    Plots metrics
    drop duplicate epochs and keeps first value
    """
    df = map_epochs(df)

    df_main = pd.DataFrame()
    for metric in metrics:
        dft = df[df.metric == metric]
        dft.set_index("epoch", drop=True, inplace=True)
        dft = dft[["value"]].drop_duplicates(keep="first")
        dft.rename(columns={"value": metric}, inplace=True)
        dft.sort_index(inplace=True)
        df_main = pd.concat([df_main, dft], 1)
    if not df_main.empty:
        df_main.plot(figsize=(10, 10))
        plt.xlabel("Epoch")
        plt.legend(metrics)
    return df_main


def get_test_dataloader(hparams, out_size=(480, 640)):
    test_csv = hparams.test_data
    dataset = hparams.dataset if "dataset" in hparams else "hypersim"

    if dataset == "hypersim":
        test_csv = "data/test_data.csv"
        test_data = HypersimDataset(
            csv_file=test_csv, data_path="./data/", out_size=out_size
        )
    elif dataset == "nyu":
        test_data = NYUDataset(dataset_type="test", out_size=out_size)

    return DataLoader(test_data, batch_size=32, num_workers=0, shuffle=False)


def main():
    # ##########################
    # metrics_list = [
    #     ["val_mIoU", "val_acc"],
    #     # ["val_log10_err"],
    #     ["val_rel_err"],
    #     ["val_rms_err"],
    # ]
    # ##########################
    p = Path("./experiment_nyu/")
    paths = [str(pa) for pa in list(p.glob("**/events.out.tfevents*"))]

    for path in paths:
        try:
            """
            Getting and making paths
            """
            root = os.path.dirname(path)
            save_dir = os.path.join(root, "results")

            # -------------------------
            # USE WITH CAUTIOUS!!!
            # shutil.rmtree(save_dir)
            # continue
            # -------------------------

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            event_id = str(os.path.basename(path)).replace("events.out.tfevents.", "")
            """
            getting data from events.out
            """
            df = parsetf(path=path)
            keys = df.metric.unique()
            file_type = "test" if "test_loss" in keys else "train_val"
            df = map_epochs(df)
            df = pd.pivot_table(df, values="value", index="epoch", columns="metric")
            df.drop("epoch", axis=1, inplace=True)
            df_path = os.path.join(save_dir, f"{file_type}_data.csv")
            if os.path.exists(df_path):
                if file_type == "test":
                    os.remove(df_path)
                else:
                    dft = pd.read_csv(df_path)
                    if len(dft) > len(df):
                        df = dft.copy()
            df = df.round(4)
            if file_type == "test":
                df.drop("test_log10_err", axis=1, inplace=True)
            df.to_csv(df_path)
            if file_type == "train_val":
                loss_metric = ["train_loss", "val_loss"]
                if "train_loss_epoch" in keys:
                    loss_metric[0] = "train_loss_epoch"
                __buff = df[loss_metric]
                save_path = os.path.join(save_dir, f"{'-'.join(loss_metric)}.png")
                if not __buff.empty:
                    __buff.plot()
                    plt.savefig(save_path)
                plt.close()

                """
                Going over metrics and plotting
                """
                for metrics in metrics_list:
                    try:
                        __buff = df[metrics]
                        save_path = os.path.join(save_dir, f"{'-'.join(metrics)}.png")
                        if not __buff.empty:
                            __buff.plot()
                            plt.savefig(save_path)
                        plt.close()
                    except:
                        pass
        except:
            pass

    generate_figures()


def generate_figures():
    p = Path("./experiment_nyu/")
    paths = [str(pa) for pa in list(p.glob("**/*.ckpt"))]

    for path in paths:
        # print(path)
        try:
            root = os.path.dirname(os.path.dirname(path))
            hparams_file = os.path.join(root, "hparams.yaml")

            try:
                with open(hparams_file) as f:
                    config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except:
                print("Failed: " + str(path))
                print("Message: missing hparams.yaml")
                continue
            hparams = dict_to_args(config_dict)

            # if not "recon" in hparams.model_type:
            #     continue

            output_res = (hparams.img_size_h, hparams.img_size_w)

            test_loader = get_test_dataloader(hparams, out_size=output_res)
            test_imgs, true_depth, true_seg = next(iter(test_loader))
            img_indices = [0, 18, 23, 30]

            pretrained_model = Experiment.load_from_checkpoint(path, hparams=hparams)
            pretrained_model.eval()
            
            if "recon" in hparams.model_type:
                seg_out, depth_out, recon_out = pretrained_model(test_imgs[img_indices, :, :, :])

            else:
                seg_out, depth_out = pretrained_model(test_imgs[img_indices, :, :, :])
            seg_pred = seg_out.argmax(1)
            # break
            for out, img_idx in enumerate(img_indices):
                # print(recon_out[out, :, :, :].shape)
                save_results(
                    torch.clone(test_imgs[img_idx]),
                    seg_pred[out],
                    depth_out[out],
                    true_depth[img_idx],
                    true_seg[img_idx],
                    root
                    + f'/results/{os.path.basename(path).strip(".ckpt")}_{output_res[1]}/',
                    f"test_{img_idx}",
                    recon_out=recon_out[out, :,:,:]
                )
            print("Success: " + str(path))
            del pretrained_model, seg_out, depth_out, seg_pred
        except Exception as e:
            print("Failed: " + str(path))
            print("Message: " + str(e))


if __name__ == "__main__":
    main()
