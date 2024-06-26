import torch 
import rad_polygon_mask as ray_mask

import pdb 
import time

if __name__ == "__main__":
    # device = 'cpu'
    device = "cuda:0"
    batch_size = 8
    # 64 * 4
    gt_mask = torch.zeros(4, 1, 512, 512).to(device)
    gt_mask[:, :, 50:290, 50:290] = 1.
    # pdb.set_trace()
    print(gt_mask.sum())
    sector_num = 4
    start = time.time()
    # while True:
    output = ray_mask.polygon_mask(gt_mask, sector_num, False)
    print("time cost {:.10f}".format(time.time() - start))
    # pdb.set_trace()
    # output[0][9]
    print(output[0][0])
    # pdb.set_trace()