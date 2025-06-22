from pathlib import Path

import tifffile
import numpy as np
from careamics import CAREamist
from careamics.config import create_n2v_configuration
from careamics.utils.metrics import scale_invariant_psnr
from careamics.utils.metrics import avg_ssim


def train_and_predict(data_train):
    """
    Trains N2V model and then predicts results
    :param data_train: Noisy data which needs to be trained on has shape (N,H,W)
    :return: Predicted denoised images
    """
    config = create_n2v_configuration(
        experiment_name="synthetic_n2v",
        data_type="array",
        axes="SYX",
        patch_size=[64, 64],
        batch_size=32,
        num_epochs=40,
        use_n2v2=False,
    )

    careamist = CAREamist(config)

    careamist.train(
        train_source=data_train
    )

    pred = careamist.predict(source=data_train)

    return pred


def write_ssim_psnr(gt, pred):
    """
    Computes SSIM and PSNR and writes them to a file
    :param gt: Ground truth image ndarray (N,H,W)
    :param pred: Denoised image ndarray (N,H,W)
    """
    psnrs = np.zeros(gt.shape[0])
    for i in range(gt.shape[0]):
        psnrs[i] = scale_invariant_psnr(gt[i], pred[i])

    ssim = avg_ssim(gt, pred)

    output_path = "./results/n2v_results.txt"
    with open(output_path, "w") as f:
        f.write(f"PSNR: {np.mean(psnrs):.3f} +/- {np.std(psnrs):.3f}\n")
        f.write(f"SSIM: {ssim[0]:.3f} +/- {ssim[1]:.3f}\n")


gt_folder = Path("./data/gt_data/")
noisy_folder = Path("./data/noisy_data/")

gt_files = sorted(gt_folder.glob("*.tif"))
noisy_files = sorted(noisy_folder.glob("*.tif"))

gt_data = np.concatenate([tifffile.imread(str(file)) for file in gt_files], axis=0)
noisy_data = np.concatenate([tifffile.imread(str(file)) for file in noisy_files], axis=0)

prediction = train_and_predict(noisy_data)
prediction = np.array([pred.squeeze() for pred in prediction])
write_ssim_psnr(gt_data, prediction)
