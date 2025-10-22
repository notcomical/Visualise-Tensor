import os
os.environ["FFMPEG_BINARY"] = r"C:\tools\ffmpeg\bin\ffmpeg.exe"
os.environ["FFPROBE_BINARY"] = r"C:\tools\ffmpeg\bin\ffprobe.exe"

from manim import *
import torch
import numpy as np

def to_np(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.array(x, dtype=np.float64)

def ensure_2cols(arr):
    a = to_np(arr)
    if a.ndim == 0:
        return a.reshape(1, -1)
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a

class TensorVectorsScene(Scene):
    def construct(self):
        v_single = torch.tensor([2.5, 1.2], dtype=torch.float32)
        V = torch.tensor([[2.0, 1.0], [-1.0, 2.0], [3.0, -2.0]], dtype=torch.float32)
        origins = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]], dtype=torch.float32)

        def draw_vectors(vecs, origs=None, colors=None, label_prefix="v", shift=None):
            vecs_np = ensure_2cols(vecs)
            N = vecs_np.shape[0]
            if origs is None:
                origs_np = np.zeros((N, 2), dtype=np.float64)
            else:
                origs_np = ensure_2cols(origs)
                if origs_np.shape[0] != N:
                    if origs_np.shape[0] == 1:
                        origs_np = np.repeat(origs_np, N, axis=0)
                    else:
                        raise ValueError("Number of origins must match number of vectors")
            shift_np = np.array(shift if shift is not None else [0.0, 0.0], dtype=np.float64)
            group = VGroup()
            for i in range(N):
                ox, oy = float(origs_np[i, 0]), float(origs_np[i, 1])
                vx, vy = float(vecs_np[i, 0]), float(vecs_np[i, 1])
                start = np.array([ox, oy, 0.0]) + np.append(shift_np, 0.0)
                end = np.array([ox + vx, oy + vy, 0.0]) + np.append(shift_np, 0.0)
                color = colors[i] if (colors is not None and i < len(colors)) else BLUE
                arrow = Arrow(start, end, buff=0, max_tip_length_to_length_ratio=0.15, color=color)
                label = Text(f"{label_prefix}_{i}", font_size=24).scale(0.6).next_to(arrow.get_end(), RIGHT, buff=0.1)
                coord = Text(f"({vx:.2f},{vy:.2f})", font_size=18).next_to(arrow.get_end(), UR, buff=0.05)
                group.add(arrow, label, coord)
            self.play(Create(group))
            return group

        axes = NumberPlane(
            x_range=[-6, 6, 1],
            y_range=[-6, 6, 1],
            x_length=10,
            y_length=10,
            background_line_style={"stroke_opacity": 0.2},
        ).to_edge(LEFT, buff=0.5)
        self.add(axes)

        self.wait(0.5)
        grp1 = draw_vectors(v_single, origs=None, colors=[YELLOW], label_prefix="v")
        self.wait(1.0)
        self.play(FadeOut(grp1))

        colors = [RED, GREEN, BLUE]
        self.wait(0.3)
        grp2 = draw_vectors(V, origs=None, colors=colors, label_prefix="v")
        self.wait(1.0)

        self.play(FadeOut(grp2))
        self.wait(0.2)
        grp3 = draw_vectors(V, origs=origins, colors=colors, label_prefix="w")
        self.wait(1.0)

        self.play(FadeOut(grp3))
        self.wait(0.3)

        v1 = V[0].unsqueeze(0)
        v2 = V[1].unsqueeze(0)
        g1 = draw_vectors(v1, origs=None, colors=[RED], label_prefix="v1")
        self.wait(0.6)
        tip_of_v1 = to_np(v1[0])
        g2 = draw_vectors(v2, origs=tip_of_v1.reshape(1, 2), colors=[GREEN], label_prefix="v2")
        self.wait(0.6)
        resultant = v1 + v2
        g3 = draw_vectors(resultant, origs=None, colors=[BLUE], label_prefix="r")
        self.wait(1.0)

        self.wait(2.0)
