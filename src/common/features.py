import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from src.common.data import load_data
from src.common.paths import DATA, DB_TEST, DB_TRAIN, ATTRS_TRAIN

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
def extract_corpus(split='test', batch_size=256, out_path=None,
                   with_labels=False, labels_path=None, log_every=50):
    """Estrae le feature visive di tutto il corpus. Riga r == celeba[r].

    shuffle=False + num_workers=0 => l'ordine del loader == celeba[0,1,2,...],
    quindi riga r delle feature corrisponde a celeba[r]. INVARIANTE SACRO.

    with_labels=True raccoglie anche la matrice 40-attributi di CelebA, allineata
    riga-per-riga alle feature (serve al sampler self-supervised del training Φ).
    Se labels_path è dato, la salva. Ritorna F oppure (F, L) se with_labels.
    """
    dataset = load_data(str(DATA), split, download=False)
    if with_labels:
        # tieni immagine E target (le 40 label binarie); collate scartava il target
        collate = lambda b: ([x[0] for x in b], torch.stack([x[1] for x in b]))
    else:
        collate = lambda b: [x[0] for x in b]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate)

    feats, labels = [], []
    n_batches = len(loader)
    for i, batch in enumerate(loader):
        imgs, lbls = batch if with_labels else (batch, None)
        z, _ = encode_data(images=imgs)
        feats.append(z.cpu())
        if with_labels:
            labels.append(lbls)
        if log_every and (i % log_every == 0 or i == n_batches - 1):
            print(f"batch {i + 1}/{n_batches}")

    F = torch.cat(feats, dim=0)
    print(f"Feature matrix shape: {tuple(F.shape)}")
    if out_path:
        torch.save(F, out_path)
        print(f"Salvato in {out_path}")

    if with_labels:
        L = torch.cat(labels, dim=0)              # [N, 40], allineato a F
        print(f"Label matrix shape: {tuple(L.shape)}")
        if labels_path:
            torch.save(L, labels_path)
            print(f"Salvato in {labels_path}")
        return F, L
    return F


if __name__ == "__main__":
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else 'test'
    if split == 'train':
        # Train corpus per il training di Φ (Solution B). Full split, ~162k img.
        extract_corpus(split='train', batch_size=256, out_path=str(DB_TRAIN),
                       with_labels=True, labels_path=str(ATTRS_TRAIN))
    else:
        extract_corpus(split='test', out_path=str(DB_TEST))
