import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange
import os
import re
from typing import List, Tuple
from PIL import Image
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm

from dinov2.eval.setup import build_model_for_eval
from dinov2.configs import load_and_merge_config
from segment_anything import sam_model_registry, SamPredictor


# Configurations
IMG_SIZE = (448, 448)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def get_transformations() -> Tuple[T.Compose, T.Compose]:
    """Get image and mask transformations."""
    transform_img = T.Compose([
        T.Resize(IMG_SIZE), 
        T.ToTensor(),
        T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    transform_mask = T.Compose([T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.NEAREST), T.ToTensor()])
    return transform_img, transform_mask


def load_images_from_dir(directory: str, transform: T.Compose) -> List[torch.Tensor]:
    """Loads and processes images from a directory."""
    files = sorted(os.listdir(directory), key=lambda x: int(re.search(r'\d+', x).group()))
    return [transform(Image.open(os.path.join(directory, file))).cuda() for file in files]


def load_supports_and_queries(spt_imgs_dir: str, spt_msks_dir: str, qry_imgs_dir: str) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """Loads and processes support and query images and masks."""
    transform_img, transform_mask = get_transformations()
    spt_imgs = [img.unsqueeze(0) for img in load_images_from_dir(spt_imgs_dir, transform_img)]
    fg_msks = [transform_mask(Image.open(os.path.join(spt_msks_dir, file)).convert('L')).cuda().bool() for file in sorted(os.listdir(spt_msks_dir), key=lambda x: int(re.search(r'\d+', x).group()))]
    bg_msks = [(~mask).int() for mask in fg_msks]
    qry_imgs = torch.stack(load_images_from_dir(qry_imgs_dir, transform_img))
    return spt_imgs, fg_msks, bg_msks, qry_imgs


def dino_backbone_embed(image_tensor: torch.Tensor, n_patch: int = 32, model_scale: str = 'vits14') -> torch.Tensor:
    """Passes image tensor through DINO backbone model and returns embeddings."""
    config_path = f'eval/{model_scale}_reg4_pretrain'
    checkpoint_path = f'./dinov2/checkpoints/dinov2_{model_scale}_reg4_pretrain.pth'
    conf = load_and_merge_config(config_path)
    model = build_model_for_eval(conf, checkpoint_path).cuda()
    with torch.no_grad():
        embeddings = model.forward_features(image_tensor)['x_norm_patchtokens']
    return rearrange(embeddings, 'n (a b) c -> n c a b', a=n_patch, b=n_patch)


def get_masked_features(fts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Applies mask to features and returns masked features."""
    fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
    return torch.sum(fts * mask[None, ...], dim=(2, 3)) / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)


def get_prototype(fts: List[torch.Tensor]) -> torch.Tensor:
    """Returns the prototype from the given features."""
    return sum(fts) / len(fts)


def cal_dist(fts: torch.Tensor, prototype: torch.Tensor, scaler: int = 20) -> torch.Tensor:
    """Calculates cosine similarity between features and prototype."""
    return F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler


def load_query_images(qry_imgs_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    """Loads and returns original query images and paths."""
    qry_img_list = sorted(os.listdir(qry_imgs_dir))
    transform_img = T.Resize(IMG_SIZE)
    qry_imgs = [np.array(transform_img(Image.open(os.path.join(qry_imgs_dir, qry_file)))) for qry_file in qry_img_list]
    return qry_imgs, qry_img_list


def get_query_similarity(spt_imgs_dir: str, spt_msks_dir: str, qry_imgs_dir: str) -> Tuple[List[np.ndarray], List[str], np.ndarray]:
    """Calculates similarity between support and query images."""
    supp_imgs, fore_mask, back_mask, qry_imgs = load_supports_and_queries(spt_imgs_dir, spt_msks_dir, qry_imgs_dir)
    n_shots = len(supp_imgs)
    imgs_concat = torch.cat([torch.cat(supp_imgs, dim=0), qry_imgs], dim=0)
    img_fts = dino_backbone_embed(imgs_concat)
    
    supp_fts = rearrange(img_fts[:n_shots], 'sh c h w -> sh 1 c h w').cuda()
    qry_fts = rearrange(img_fts[n_shots:], 'b c h w -> 1 b c h w').cuda()
    fore_mask = rearrange(torch.stack(fore_mask, dim=0), 'sh b h w -> sh b h w').cuda() 
    back_mask = rearrange(torch.stack(back_mask, dim=0), 'sh b h w -> sh b h w').cuda()
    
    distances = []
    for epi in tqdm(range(qry_fts.shape[1]), desc='Calculating similarity'):
        fg_prototype = get_prototype([get_masked_features(supp_fts[shot], fore_mask[shot]) for shot in range(n_shots)])
        bg_prototype = get_prototype([get_masked_features(supp_fts[shot], back_mask[shot]) for shot in range(n_shots)])
        dist = [cal_dist(qry_fts[:, epi], prototype) for prototype in [bg_prototype, fg_prototype]]
        distances.append(torch.stack(dist, dim=1).squeeze())

    sim_map = (torch.stack(distances)[:, 1] - torch.stack(distances)[:, 0]).cpu().numpy()
    qry_orgn_imgs, qry_orgn_path = load_query_images(qry_imgs_dir)
    resized_sim_map = np.array([cv2.resize(sim_map[i], qry_orgn_imgs[i].shape[:-1], interpolation=cv2.INTER_LINEAR) for i in range(len(qry_orgn_imgs))])
    
    return qry_orgn_imgs, qry_orgn_path, resized_sim_map


def get_tumor_region_prompt(similarity_map: np.ndarray, threshold: float = 0.85) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """Extracts tumor regions and their centers from the similarity map."""
    regions, centers = [], []
    for matrix in similarity_map:
        matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
        binary_matrix = matrix > threshold
        labeled_matrix, num_features = ndimage.label(binary_matrix)
        largest_region = labeled_matrix == np.argmax(ndimage.sum(binary_matrix, labeled_matrix, range(num_features + 1)))
        centers.append(ndimage.center_of_mass(largest_region))
        regions.append(largest_region)
    return regions, centers


def generate_masks(qry_orgn_imgs: List[np.ndarray], 
                   qry_orgn_path: List[str], 
                   ctrs: List[Tuple[float, float]],
                   save_mask_path: str):
    """Generates masks using SAM and saves the results."""
    sam_checkpoint = "D:/GithubProjects/segment-anything-main/weights/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).cuda()
    predictor = SamPredictor(sam)
    os.makedirs(save_mask_path, exist_ok=True)

    for i, image in enumerate(tqdm(qry_orgn_imgs, desc='Generating masks')):
        point = [ctrs[i][1], ctrs[i][0]]
        predictor.set_image(image)

        masks, _, logits = predictor.predict(
            point_coords=np.array([point]), 
            point_labels=np.array([1]), 
            multimask_output=False
        )
        
        mask_input = logits[np.argmax(_), :, :]
        mask, _, _ = predictor.predict(
            point_coords=np.array([point, [1, 1]]), 
            point_labels=np.array([1, 0]), 
            mask_input=mask_input[None, :, :], 
            multimask_output=False
        )

        Image.fromarray(mask[0]).save(os.path.join(save_mask_path, f'{qry_orgn_path[i]}_mask.jpg'))


if __name__ == '__main__':
    support_images_dir = './dataset/supports'
    support_masks_dir = './dataset/supports_masks'
    query_images_dir = './dataset/demo'
    save_mask_path = './dataset/demo_mask'

    qry_orgn_imgs, qry_orgn_path, sim_map = get_query_similarity(support_images_dir, support_masks_dir, query_images_dir)
    regions, centers = get_tumor_region_prompt(sim_map)
    generate_masks(qry_orgn_imgs, qry_orgn_path, centers, save_mask_path)
