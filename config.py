import os
import math


class Config():
    def __init__(self) -> None:
        # ==== General settings ====
        self.batch_size = 4
        self.compile = False
        self.mixed_precision = 'no'
        self.SDPA_enabled = True

        # ==== Paths ====
        # Keep everything local in the repo
        self.sys_home_dir = os.path.abspath(".")
        self.data_root_dir = os.path.join(self.sys_home_dir, "data")

        # Your dataset
        self.task = "DrawerForeground"
        self.training_set = "train"
        self.testsets = "val"

        # ==== Data settings ====
        self.size = (512, 512)
        self.dynamic_size = None
        self.background_color_synthesis = False
        self.precisionHigh = True
        self.load_all = False
        self.num_workers = 0  # safer for small custom dataset

        # ==== Model ====
        self.model = "BiRefNet"
        self.bb = "resnet50"
        self.freeze_bb = False
        self.ms_supervision = True
        self.out_ref = True
        self.dec_ipt = True
        self.dec_ipt_split = True
        self.cxt_num = 3
        self.mul_scl_ipt = "cat"
        self.dec_att = "ASPP"
        self.squeeze_block = "BasicDecBlk_x1"
        self.dec_blk = "BasicDecBlk"
        self.verbose_eval = True


        # required extras
        self.lat_blk = "BasicLatBlk"
        self.dec_channels_inter = "fixed"
        self.auxiliary_classification = False

        # ==== Decoder channel layout ====
        self.lateral_channels_in_collection = {
            'resnet50': [2048, 1024, 512, 256],
        }[self.bb]
        if self.mul_scl_ipt == "cat":
            self.lateral_channels_in_collection = [c * 2 for c in self.lateral_channels_in_collection]
        self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num:] if self.cxt_num else []

        # ==== Training ====
        self.finetune_last_epochs = -20
        self.lr = 1e-4
        self.optimizer = "AdamW"
        self.lr_decay_epochs = [1e5]
        self.lr_decay_rate = 0.5
        self.preproc_methods = ["flip", "enhance", "rotate", "crop"]

        # ==== Loss ====
        self.lambdas_pix_last = {
            'bce': 30,
            'iou': 0.5,
            'mae': 0,
            'mse': 0,
            'triplet': 0,
            'reg': 0,
            'ssim': 10,
            'cnt': 0,
            'structure': 0,
        }
        self.lambdas_cls = {'ce': 5.0}

        # ==== Runtime ====
        self.device = 0  # 'cuda:0'
        self.batch_size_valid = 1
        self.rand_seed = 7

        # ==== Fallback for checkpoint save frequency ====
        self.save_last = 1
        self.save_step = 1


# Return task for choosing settings in shell scripts.
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Only choose one argument to activate.')
    parser.add_argument('--print_task', action='store_true', help='print task name')
    args = parser.parse_args()

    config = Config()
    if args.print_task:
        print(config.task)
