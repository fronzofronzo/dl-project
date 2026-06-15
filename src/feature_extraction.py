import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from load_data import load_data

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


@torch.no_grad()
def encode_data(images=None, texts=None):
    """Encode immagini (PIL) e/o testi con CLIP ViT-B/32 (HF).
    Ritorna (images_z, texts_z), L2-normalizzati. None se input mancante."""
    images_z = texts_z = None

    if images is not None:
        img_in = processor(images=images, return_tensors="pt").to(device)
        # transformers 5.x: get_image_features -> output object; pooler_output = embed 512
        images_z = model.get_image_features(**img_in).pooler_output.float()
        images_z = images_z / images_z.norm(dim=-1, keepdim=True)

    if texts is not None:
        txt_in = processor(text=texts, return_tensors="pt", padding=True).to(device)
        texts_z = model.get_text_features(**txt_in).pooler_output.float()
        texts_z = texts_z / texts_z.norm(dim=-1, keepdim=True)

    return images_z, texts_z


@torch.no_grad()
def extract_corpus(split='test', batch_size=256, out_path=None):
    """Estrae le feature visive di tutto il corpus. Riga r == celeba[r]."""
    dataset = load_data('data', split, download=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=lambda b: [x[0] for x in b])

    feats = []
    for imgs in loader:
        z, _ = encode_data(images=imgs)
        feats.append(z.cpu())

    F = torch.cat(feats, dim=0)
    print(f"Feature matrix shape: {tuple(F.shape)}")
    if out_path:
        torch.save(F, out_path)
        print(f"Salvato in {out_path}")
    return F


if __name__ == "__main__":
    extract_corpus(split='test', out_path='data/clip_features_test.pt')
