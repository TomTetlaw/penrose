import os, re
import numpy as np
import imageio.v2 as imageio
from scipy.ndimage import sobel
from PIL import Image
import sys
TARGET_SIZE = 512
PENROSE_PATH = "F:/penrose"
def load_tex(path, fallback_value=1.0):
    if not path or not os.path.exists(path):
        print(f"could not load file {path}")
        os._exit(1)
    img = imageio.imread(path).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = img[..., None]
    elif img.shape[2] > 3:
        img = img[..., :3]
    return img
def resize_tex(tex, size=(TARGET_SIZE,TARGET_SIZE)):
    img = Image.fromarray((np.clip(tex,0,1)*255).astype(np.uint8))
    tex_resized = np.array(img.resize(size, Image.BILINEAR)).astype(np.float32)/255.0
    if tex_resized.ndim == 2:
        tex_resized = tex_resized[..., None]
    return tex_resized
def save_tex(path, img):
    img = np.clip(img, 0, 1)
    imageio.imwrite(path, (img * 255).astype(np.uint8))
    print(f"saved {path}")
def height_to_normal(height, strength=1.0):
    if height.ndim == 3:
        height = height[..., 0]
    gx = sobel(height, axis=1)
    gy = sobel(height, axis=0)
    n = np.stack((-gx, -gy, np.ones_like(height) / strength), axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
    return (n * 0.5 + 0.5).astype(np.float32)
def combine_normals(n1, n2):
    n1 = n1 * 2.0 - 1.0
    n2 = n2 * 2.0 - 1.0
    n = np.zeros_like(n1)
    n[..., 0] = n1[..., 0] + n2[..., 0]
    n[..., 1] = n1[..., 1] + n2[..., 1]
    n[..., 2] = n1[..., 2] * n2[..., 2]
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
    return n * 0.5 + 0.5
def process_material(name, paths):
    print(f"→ Processing {name}")
    albedo    = resize_tex(load_tex(paths.get('albedo'), 1.0))
    ao        = resize_tex(load_tex(paths.get('ao'), 1.0))
    height    = resize_tex(load_tex(paths.get('height'))) if 'height' in paths else None
    metallic  = resize_tex(load_tex(paths.get('metallic'), 0.0))
    normal    = resize_tex(load_tex(paths.get('normal'), 0.5))
    roughness = resize_tex(load_tex(paths.get('roughness'), 0.0))
    diffuse = albedo[..., :3] * ao
    diffuse = np.power(diffuse, 1.2)
    grey = np.mean(diffuse, axis=-1, keepdims=True)
    diffuse = diffuse * 0.8 + grey * 0.2
    spec = (1.0 - 0.8 * roughness)
    spec *= ao
    spec_color = spec * (1 - roughness) + albedo[..., :3] * roughness
    spec_gray = 0.299 * spec_color[..., 0] + 0.587 * spec_color[..., 1] + 0.114 * spec_color[..., 2]
    spec_gray_rgb = np.dstack([spec_gray, spec_gray, spec_gray])
    final_normal = normal[..., :3]
    if height is not None:
        h_norm = height_to_normal(height, strength=1.0)
        final_normal = combine_normals(final_normal, h_norm)
    save_tex(f"{PENROSE_PATH}/build/textures/{name}.png", diffuse)
    save_tex(f"{PENROSE_PATH}/build/textures/{name}_norm.png", final_normal)
    save_tex(f"{PENROSE_PATH}/build/textures/{name}_spec.png", spec_gray_rgb)
    print(f"   ✔ {name}.png, {name}_norm.png, {name}_spec.png ({TARGET_SIZE}×{TARGET_SIZE})")
def main():
    search_dir = sys.argv[2]
    print(os.listdir(search_dir))
    files = [os.path.join(search_dir, f) for f in os.listdir(search_dir) if os.path.isfile(os.path.join(search_dir, f)) and re.search(r'\.(png|jpg|tga|tif)$', f, re.I)]
    groups = {}
    for f in files:
        lower = f.lower()
        match = re.search(r'(albedo|ao|height|metallic|normal|roughness)', lower)
        if not match:
            continue
        key = match.group(1)
        matname = sys.argv[1]
        groups.setdefault(matname, {})[key] = f
    if not groups:
        print("No matching PBR textures found.")
        return
    for name, paths in groups.items():
        process_material(name, paths)
if __name__ == "__main__":
    main()
