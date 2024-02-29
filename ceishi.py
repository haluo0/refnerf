import torch
filename = 'D:/project/scratch/jcu/ai_group/xsc_data/dataset/model/model_255000.pth'
to_load = torch.load(filename)
# optimizer.load_state_dict(to_load['optimizer'])


for key in to_load['optimizer'].items():
    print(key)