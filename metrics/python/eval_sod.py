import os
import cv2
import metrics as M
from multiprocessing import Pool

def cal_metrics(pred_root, gt_root, dataset,log="log.txt"):
    eval_folder = pred_root.split(os.path.sep)[-1]
    gt_dataset = os.path.join(gt_root, dataset)
    pred_root = os.path.join(pred_root, dataset)

    FM = M.Fmeasure()
    WFM = M.WeightedFmeasure()
    SM = M.Smeasure()
    EM = M.Emeasure()
    MAE = M.MAE()


    f = open(log, "a")

    gt_root = os.path.join(gt_dataset, 'mask')

    gt_name_list = sorted(os.listdir(pred_root))

    for gt_name in gt_name_list:
        gt_path = os.path.join(gt_root, gt_name)
        pred_path = os.path.join(pred_root, gt_name)

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        MAE.step(pred=pred, gt=gt)

    fm = FM.get_results()['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']

    results = f"Method:{eval_folder.ljust(8)}; " \
              f"Dataset:{dataset.ljust(10)}; " \
              f"Fm:{str(fm['adp'].round(3)).ljust(5,'0')} ; " \
              f"MAE:{str(mae.round(3)).ljust(5,'0')} ; " \
              f"wFm:{str(wfm.round(3)).ljust(5,'0')} ; " \
              f"Sm:{str(sm.round(3)).ljust(5,'0')} ; " \
              f"mEm:{str('-' if em['adp'] is None else em['adp'].mean().round(3)).ljust(5,'0')} ;"

    print(results)
    f.write(f"{results}\n")

    f.close()
    return {dataset: mae}


def eval_all_datasets(pred_root, gt_root, datasets,log="metrics.txt"):
    pool = Pool(processes=5)
    metrics = pool.starmap(cal_metrics, [[pred_root, gt_root, dataset,log] for dataset in datasets])
    return metrics


if __name__ == '__main__':
    eval_all_datasets(pred_root=rf"path/to/pred_dir",
                      gt_root=r"path/to/test",
                      datasets=['DUT-OMRON', 'DUTS-TE', 'ECSSD', 'HKU-IS', 'PASCAL-S'],
                      log="metrics.txt")