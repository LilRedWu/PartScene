import torch
import torch.nn as nn
from torch.optim import AdamW

class Florence2Mock(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        )
        self.decoder = nn.Conv2d(hidden_dim, 1, 1)  # Vision decoder
        self.text_embed = nn.Embedding(1000, hidden_dim)  # Mock text embedding
    
    def forward(self, image, text_query):
        B, C, H, W = image.shape
        img_feat = self.encoder(image)
        text_feat = self.text_embed(torch.randint(0, 1000, (1,)).to(image.device)).mean(dim=0)
        fused = img_feat + text_feat.view(1, -1, 1, 1)
        mask = torch.sigmoid(self.decoder(fused))
        return mask

def segment_2d_parts(views, text_query, ground_truth_masks=None, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Florence2Mock().to(device)
    optimizer = AdamW(model.decoder.parameters(), lr=1e-4)  # Only fine-tune decoder
    
    if ground_truth_masks:
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for view, gt in zip(views, ground_truth_masks):
                view_tensor = torch.tensor(view, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                gt_tensor = torch.tensor(gt, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                
                optimizer.zero_grad()
                pred = model(view_tensor, text_query)
                loss = -torch.log(torch.softmax(pred, dim=1) * gt_tensor + 1e-6).sum()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(views):.4f}")
    
    model.eval()
    masks = []
    with torch.no_grad():
        for view in views:
            view_tensor = torch.tensor(view, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            mask = model(view_tensor, text_query).squeeze().cpu().numpy()
            masks.append(mask)
    return masks