import numpy as np
from PIL import Image

def pca_color_aug(img):
    img = img.detach().numpy()
    og_img = img.astype(float).copy()
    imge = img/255.0

    img = imge.reshape(-1, 3)
    img_centered = img - np.mean(img, axis=0)

    img_cov = np.cov(img_centered, rowvar=False)

    eig_vals, eig_vecs = np.linalg.eigh(img_cov)
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    m1 = np.column_stack((eig_vecs))

    m2 = np.zeros((3, 1))
    alpha = np.random.normal(0, 0.1)

    m2[:, 0] = alpha * eig_vals[:]
    add_vec = np.matrix(m1) * np.matrix(m2)
    
    for i in range(3):
        og_img[..., i] += add_vec[i]
    og_img = og_img * 255
    og_img = np.clip(og_img, 0.0, 255.0)
    og_img = og_img.astype(np.uint8)

    return og_img