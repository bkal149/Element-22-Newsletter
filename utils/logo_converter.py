"""
Utility to convert and optimize the Element22 logo
"""
from PIL import Image
import os

def optimize_logo():
    """Optimize logo for web display"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logo_path = os.path.join(base_dir, "assets", "element22_logo.png")
    
    if os.path.exists(logo_path):
        # Open and optimize
        img = Image.open(logo_path)
        
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Resize if too large (max height 120px)
        if img.height > 120:
            ratio = 120 / img.height
            new_size = (int(img.width * ratio), 120)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save optimized version
        optimized_path = os.path.join(base_dir, "assets", "element22_logo_optimized.png")
        img.save(optimized_path, 'PNG', optimize=True)
        
        print(f"✓ Logo optimized and saved to: {optimized_path}")
        print(f"  Original size: {os.path.getsize(logo_path) / 1024:.2f} KB")
        print(f"  Optimized size: {os.path.getsize(optimized_path) / 1024:.2f} KB")
    else:
        print(f"✗ Logo not found at: {logo_path}")

if __name__ == "__main__":
    optimize_logo()