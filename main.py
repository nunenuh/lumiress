import typer
from lumiress.infer import mirnetv2_infer
from pathlib import Path
from lumiress import io

def main(src: str = '', dst: str = '', mode: str = 'lowlight', device: str = "cpu"):
    if mode=='lowlight':
        infer = mirnetv2_infer.lowlight_enchance(device=device)
    elif mode=='contrast':
        infer = mirnetv2_infer.contrast_enhance(device=device)
    elif mode=='denoise':
        infer = mirnetv2_infer.real_denoising(device=device)
    elif mode=='sr':
        infer = mirnetv2_infer.super_resolution(device=device)
    else:
        raise ValueError(f"Mode {mode} not supported")
    
    src: Path = Path(src)
    
    if dst is not None:
        dst: Path = Path(dst)
    else:
        base_path = src.parent
        fname = src.stem
        fname = f'{fname }_{mode}{src.suffix}'
        dst = Path(base_path).joinpath(fname)
        
    
    if src.exists() and src.is_file():
        result = infer.restore(src)
        if not dst.exists():
            io.save_img(dst, result)
        else:
            raise FileExistsError(f"File {dst} already exists")
            
    


if __name__ == "__main__":
    typer.run(main)