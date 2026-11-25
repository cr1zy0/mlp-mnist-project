import argparse
import struct
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk


SIZE = 280
BRUSH = 24


def prepare_digit(image):
    """Convert a white-on-black drawing into a centered 28x28 MNIST image."""
    bbox = image.getbbox()
    if bbox is None:
        raise ValueError("Draw a digit first.")

    digit = image.crop(bbox)
    scale = 20 / max(digit.size)
    width, height = (max(1, round(d * scale)) for d in digit.size)
    digit = digit.resize((width, height), Image.Resampling.LANCZOS)

    result = Image.new("L", (28, 28))
    pixels = list(digit.getdata())
    mass = sum(pixels)
    cx = sum((i % width) * p for i, p in enumerate(pixels)) / mass
    cy = sum((i // width) * p for i, p in enumerate(pixels)) / mass
    x = max(0, min(28 - width, round(13.5 - cx)))
    y = max(0, min(28 - height, round(13.5 - cy)))
    result.paste(digit, (x, y))
    return result


def save_idx(image, output_dir):
    if image.mode != "L" or image.size != (28, 28):
        raise ValueError("Expected a 28x28 grayscale image.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_path = output_dir / "inference-images.idx3-ubyte"
    labels_path = output_dir / "inference-labels.idx1-ubyte"

    # Big-endian magic, count, rows, columns, followed by 784 pixel bytes.
    images_path.write_bytes(struct.pack(">4I", 2051, 1, 28, 28) + image.tobytes())

    # The existing C loader requires one valid label. It is a placeholder,
    # not the real class, and must not be used to evaluate the prediction.
    labels_path.write_bytes(struct.pack(">2I", 2049, 1) + b"\x00")
    return images_path, labels_path


class DigitDrawer:
    def __init__(self, root, output_dir):
        self.root = root
        self.output_dir = output_dir
        self.image = Image.new("L", (SIZE, SIZE))
        self.draw = ImageDraw.Draw(self.image)
        self.last = None
        root.title("MNIST digit drawer")
        root.resizable(False, False)
        root.configure(padx=18, pady=18, bg="#f3f5f3")

        tk.Label(root, text="Draw one digit", font=("Segoe UI", 18, "bold"),
                 bg="#f3f5f3").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        self.canvas = tk.Canvas(root, width=SIZE, height=SIZE, bg="black", highlightthickness=0)
        self.canvas.grid(row=1, column=0)
        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.move_stroke)
        self.canvas.bind("<ButtonRelease-1>", lambda event: setattr(self, "last", None))

        panel = tk.Frame(root, bg="#f3f5f3", padx=16)
        panel.grid(row=1, column=1, sticky="n")
        tk.Label(panel, text="28 x 28 preview", bg="#f3f5f3", font=("Segoe UI", 10)).pack(pady=(0, 8))
        self.preview = tk.Label(panel, bg="black", width=112, height=112)
        self.preview.pack()
        tk.Button(panel, text="Save IDX", command=self.save, width=15).pack(pady=(16, 6))
        tk.Button(panel, text="Clear", command=self.clear, width=15).pack()
        self.status = tk.StringVar(value="Draw with the mouse, then click Save IDX.")
        tk.Label(root, textvariable=self.status, bg="#f3f5f3", wraplength=440,
                 justify="left", anchor="w").grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.update_preview()

    def stroke(self, x, y):
        x, y = max(0, min(SIZE - 1, x)), max(0, min(SIZE - 1, y))
        r = BRUSH // 2
        if self.last is not None:
            self.draw.line([self.last, (x, y)], fill=255, width=BRUSH)
            self.canvas.create_line(*self.last, x, y, fill="white", width=BRUSH, capstyle=tk.ROUND)
        self.draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.last = (x, y)
        self.update_preview()

    def start_stroke(self, event):
        self.last = None
        self.stroke(event.x, event.y)

    def move_stroke(self, event):
        self.stroke(event.x, event.y)

    def update_preview(self):
        digit = prepare_digit(self.image) if self.image.getbbox() else Image.new("L", (28, 28))
        self.photo = ImageTk.PhotoImage(digit.resize((112, 112), Image.Resampling.NEAREST))
        self.preview.configure(image=self.photo)

    def clear(self):
        self.canvas.delete("all")
        self.image.paste(0, (0, 0, SIZE, SIZE))
        self.last = None
        self.update_preview()
        self.status.set("Draw with the mouse, then click Save IDX.")

    def save(self):
        try:
            paths = save_idx(prepare_digit(self.image), self.output_dir)
            self.status.set(f"Saved: {paths[0].name} and {paths[1].name}")
            print(*(str(path.resolve()) for path in paths), sep="\n")
        except (ValueError, OSError) as error:
            self.status.set(str(error))


def main():
    parser = argparse.ArgumentParser(description="Draw a digit and export one MNIST IDX sample.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = tk.Tk()
    DigitDrawer(root, args.output_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
