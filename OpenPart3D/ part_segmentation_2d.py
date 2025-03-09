import torch
import torch.nn as nn
from torch.optim import Adam

class Florence2Mock(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_decoder = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.other_components = nn.Identity()  # frozen part
    
    def forward(self, image, text_query):
        x = self.other_components(image)
        masks = torch.sigmoid(self.vision_decoder(x))
        return masks

def segment_2d_parts(views, text_query, ground_truth_masks=None):
    model = Florence2Mock()
    optimizer = Adam(model.vision_decoder.parameters(), lr=0.001)
    
    if ground_truth_masks:  # fine-tuning mode
        model.train()
        for epoch in range(5):  # few epochs for demo
            for view, gt_mask in zip(views, ground_truth_masks):
                optimizer.zero_grad()
                pred_mask = model(torch.tensor(view).permute(2, 0, 1).unsqueeze(0).float(), text_query)
                loss = -torch.log(torch.softmax(pred_mask, dim=-1) * gt_mask).sum()
                loss.backward()
                optimizer.step()
    else:
        model.eval()
    
    masks = []
    with torch.no_grad():
        for view in views:
            mask = model(torch.tensor(view).permute(2, 0, 1).unsqueeze(0).float(), text_query)
            masks.append(mask.squeeze().numpy())
    return masks