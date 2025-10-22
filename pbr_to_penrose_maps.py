import os, re
import numpy as np
import imageio.v2 as imageio
from scipy.ndimage import sobel
from PIL import Image
import sys

TARGET_SIZE = 2048
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

def process_material(name, paths):
    print(f"→ Processing {name}")
    
    albedo    = load_tex(paths.get('albedo'), 1.0)
    ao        = load_tex(paths.get('ao'), 1.0)
    roughness = load_tex(paths.get('roughness'), 0.5)
    metallic  = load_tex(paths.get('metallic'), 0.5)
    normal    = load_tex(paths.get('normal'), 0.5)

    aorm = np.dstack([
        ao[..., 0] if ao.shape[2] > 0 else np.ones_like(roughness),
        roughness[..., 0] if roughness.shape[2] > 0 else np.zeros_like(ao),
        metallic[..., 0] if metallic.shape[2] > 0 else np.zeros_like(ao)
    ])

    save_tex(f"{PENROSE_PATH}/build/textures/{name}.png", albedo)
    save_tex(f"{PENROSE_PATH}/build/textures/{name}_norm.png", normal)
    save_tex(f"{PENROSE_PATH}/build/textures/{name}_aorm.png", aorm)

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <matname> <texture_folder>")
        return

    print("hello")

    search_dir = sys.argv[2]
    matname = sys.argv[1]

    files = [os.path.join(search_dir, f) for f in os.listdir(search_dir)
             if os.path.isfile(os.path.join(search_dir, f)) and re.search(r'\.(png|jpg|tga|tif)$', f, re.I)]

    paths = {}
    for f in files:
        lower = f.lower()
        match = re.search(r'(albedo|ao|height|metallic|normal|roughness)', lower)
        if match:
            key = match.group(1)
            paths[key] = f

    if not paths:
        print("No matching PBR textures found.")
        return

    process_material(matname, paths)

if __name__ == "__main__":
    main()
