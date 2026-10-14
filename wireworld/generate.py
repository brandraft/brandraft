"""Render the BRANDRAFT Wireworld banner.

Each letter outline is a closed loop of wire. Electrons are spaced evenly so the
spacing divides every loop length, which makes the whole animation repeat exactly
(period 24). Needs numpy and ffmpeg.

    python generate.py   # writes wireworld-light.gif and wireworld-dark.gif
"""
import subprocess
import numpy as np

FONT = {
    'B': ["####.", "#...#", "####.", "#...#", "####."],
    'R': ["####.", "#...#", "####.", "#..#.", "#...#"],
    'A': [".###.", "#...#", "#####", "#...#", "#...#"],
    'N': ["#...#", "##..#", "#.#.#", "#..##", "#...#"],
    'D': ["####.", "#...#", "#...#", "#...#", "####."],
    'F': ["#####", "#....", "####.", "#....", "#...."],
    'T': ["#####", "..#..", "..#..", "..#..", "..#.."],
}
WORD, SCALE, PAD, PX = "BRANDRAFT", 4, 2, 5
EMPTY, HEAD, TAIL, WIRE = 0, 1, 2, 3
N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
PALETTES = {  # empty, head, tail, wire
    "light": [(255, 255, 255), (17, 17, 19), (120, 120, 126), (214, 214, 218)],
    "dark": [(13, 17, 23), (255, 255, 255), (130, 136, 146), (48, 54, 61)],
}


def letter_shapes():
    pix = np.array([[c == '#' for c in ".".join(FONT[ch][r] for ch in WORD)] for r in range(5)])
    return np.pad(np.kron(pix, np.ones((SCALE, SCALE), bool)), PAD)


def outline(shape):
    """Edge cells of the shape, minus convex corners, so every wire cell has exactly two neighbours."""
    H, W = shape.shape
    inside = lambda r, c: 0 <= r < H and 0 <= c < W and shape[r, c]
    wire = np.zeros_like(shape)
    for r, c in zip(*np.nonzero(shape)):
        out = [(dr, dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)) if not inside(r + dr, c + dc)]
        corner = len(out) == 2 and out[0][0] != out[1][0] and out[0][1] != out[1][1]
        wire[r, c] = bool(out) and not corner
    return wire


def loops(wire):
    H, W = wire.shape
    nbrs = {(r, c): [(r + dr, c + dc) for dr, dc in N8
                     if 0 <= r + dr < H and 0 <= c + dc < W and wire[r + dr, c + dc]]
            for r, c in zip(*np.nonzero(wire))}
    assert all(len(v) == 2 for v in nbrs.values()), "outline is not made of simple loops"
    seen, out = set(), []
    for start in nbrs:
        if start in seen:
            continue
        loop, prev, cur = [start], start, nbrs[start][0]
        seen.add(start)
        while cur != start:
            loop.append(cur); seen.add(cur)
            prev, cur = cur, next(n for n in nbrs[cur] if n != prev)
        out.append(loop)
    return out


def step(g):
    heads = (g == HEAD).astype(np.uint8)
    count = sum(np.roll(np.roll(heads, dr, 0), dc, 1) for dr, dc in N8)
    nxt = g.copy()
    nxt[g == HEAD] = TAIL
    nxt[g == TAIL] = WIRE
    nxt[(g == WIRE) & ((count == 1) | (count == 2))] = HEAD
    return nxt


def simulate():
    wire = outline(letter_shapes())
    g = np.where(wire, WIRE, EMPTY).astype(np.uint8)
    for loop in loops(wire):
        n = len(loop)
        gap = next(x for x in (8, 6, 12, 4, n) if n % x == 0)
        for p in range(0, n, gap):
            g[loop[p]], g[loop[p - 1]] = HEAD, TAIL
    frames, seen = [g], {g.tobytes()}
    while True:
        g = step(g)
        if g.tobytes() in seen:
            assert (g == frames[0]).all(), "animation does not loop back to the start"
            return np.array(frames)
        seen.add(g.tobytes()); frames.append(g)


def write_gif(path, frames, palette):
    rgb = np.array(palette, np.uint8)[frames]
    rgb = np.repeat(np.repeat(rgb, PX, 1), PX, 2)
    grid = np.ones(rgb.shape[1:3], bool)
    grid[PX - 1::PX, :] = False; grid[:, PX - 1::PX] = False
    rgb = np.where(grid[None, ..., None], rgb, np.array(palette[0], np.uint8))
    T, H, W, _ = rgb.shape
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{W}x{H}", "-r", "12", "-i", "-", "-filter_complex",
                    "split[a][b];[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=none",
                    "-loop", "0", path], input=rgb.tobytes(), check=True)


if __name__ == "__main__":
    frames = simulate()
    print(f"period {len(frames)}")
    for name, palette in PALETTES.items():
        write_gif(f"wireworld-{name}.gif", frames, palette)
