"""URCD-YOLO Model Architecture & Ultralytics Registration.

Integrates custom PyTorch modules (BiFPN, WADown, WTConv, MSCA) into Ultralytics YOLO
engine, providing the production URCDYOLO model interface.
"""

import ast
import contextlib
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.nn import tasks
from ultralytics.nn.tasks import _SafeLoad, make_divisible

from . import modules as urcd_mod

# Flag to avoid redundant patch operations
_PATCHED = False


def register_urcd_modules() -> None:
    # This function registers URCD-YOLO custom modules into the Ultralytics parsing engine.
    # Input: None -> Intermediate: binds custom classes to tasks globals and wraps parse_model -> Output: None.
    global _PATCHED
    if _PATCHED:
        return

    # 1. Register module classes directly into tasks module namespace
    custom_classes = [
        "BiFPN_Concat2",
        "BiFPN_Concat3",
        "WADown",
        "WTConv2d",
        "Bottleneck_WTConv",
        "C3k2_WTConv",
        "MSCA",
    ]
    for name in custom_classes:
        setattr(tasks, name, getattr(urcd_mod, name))

    # 2. Wrap parse_model to support channel dimension calculation for custom blocks
    def urcd_parse_model(d: dict[str, Any], ch: int, verbose: bool = True):
        # This function parses YOLO YAML definitions supporting both standard and URCD layers.
        # Input: model dict d, input channels ch -> Intermediate: layer-by-layer parameter resolution -> Output: tuple(nn.Sequential, save_list).
        legacy = True
        max_channels = float("inf")
        nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
        reg_max = d.get("reg_max", 16)
        depth, width, _ = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
        scale = d.get("scale")
        if scales:
            if not scale:
                scale = next(iter(scales.keys()))
            depth, width, max_channels = scales[scale]

        restricted = _SafeLoad.restricted()
        if act:
            tasks.Conv.default_act = _SafeLoad.activation(act) if restricted else eval(act)

        ch = [ch]
        layers, save, c2 = [], [], ch[-1]

        base_modules = frozenset(
            {
                tasks.Classify, tasks.Conv, tasks.ConvTranspose, tasks.GhostConv, tasks.Bottleneck,
                tasks.GhostBottleneck, tasks.SPP, tasks.SPPF, tasks.C2fPSA, tasks.C2PSA, tasks.DWConv,
                tasks.Focus, tasks.BottleneckCSP, tasks.C1, tasks.C2, tasks.C2f, tasks.C3k2,
                tasks.RepNCSPELAN4, tasks.ELAN1, tasks.ADown, tasks.AConv, tasks.SPPELAN,
                tasks.C2fAttn, tasks.C3, tasks.C3TR, tasks.C3Ghost, torch.nn.ConvTranspose2d,
                tasks.DWConvTranspose2d, tasks.C3x, tasks.RepC3, tasks.PSA, tasks.SCDown,
                tasks.C2fCIB, tasks.A2C2f,
                urcd_mod.WADown, urcd_mod.C3k2_WTConv,
            }
        )
        repeat_modules = frozenset(
            {
                tasks.BottleneckCSP, tasks.C1, tasks.C2, tasks.C2f, tasks.C3, tasks.C3TR,
                tasks.C3Ghost, tasks.C3x, tasks.RepC3, tasks.C2fPSA, tasks.C2fCIB, tasks.C2PSA,
                tasks.A2C2f, tasks.C3k2, urcd_mod.C3k2_WTConv,
            }
        )

        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
            m = (
                getattr(torch.nn, m[3:])
                if m.startswith("nn.")
                else getattr(__import__("torchvision").ops, m[16:])
                if m.startswith("torchvision.ops.")
                else getattr(tasks, m) if hasattr(tasks, m)
                else getattr(urcd_mod, m) if hasattr(urcd_mod, m)
                else globals().get(m, None)
            )
            if m is None:
                raise ValueError(f"Unrecognized module '{m}' in architecture specification.")

            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

            n = n_ = max(round(n * depth), 1) if n > 1 else n

            if m in base_modules:
                c1, c2 = ch[f], args[0]
                if m is not tasks.Classify:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                args = [c1, c2, *args[1:]]
                if m in repeat_modules:
                    args.insert(2, n)
                    n = 1
                if m in (tasks.C3k2, urcd_mod.C3k2_WTConv):
                    legacy = False
            elif m in (tasks.Concat, urcd_mod.BiFPN_Concat2, urcd_mod.BiFPN_Concat3):
                c2 = sum(ch[x] for x in f)
            elif m in frozenset(
                {
                    tasks.Detect, tasks.WorldDetect, tasks.YOLOEDetect, tasks.Segment,
                    tasks.Segment26, tasks.YOLOESegment, tasks.YOLOESegment26, tasks.Pose,
                    tasks.Pose26, tasks.OBB, tasks.OBB26,
                }
            ):
                args.extend([reg_max, end2end, [ch[x] for x in f]])
                m.legacy = legacy
            elif m is urcd_mod.MSCA:
                c1 = ch[f]
                c2 = c1
                args = [c1, *args]
            else:
                c2 = ch[f]

            m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace("__main__.", "")
            m_.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)

        return torch.nn.Sequential(*layers), sorted(save)

    tasks.parse_model = urcd_parse_model
    _PATCHED = True


class URCDYOLO(YOLO):
    """URCD-YOLO model for UAV road crack detection.

    Inherits from Ultralytics YOLO and automatically wires custom architectural
    innovations (BiFPN, WADown, WTConv, MSCA).
    """

    DEFAULT_CFG = str(Path(__file__).resolve().parent / "urcd_yolo11s.yaml")

    def __init__(self, model: str | None = None, task: str = "detect", verbose: bool = True):
        # This function initializes the URCD-YOLO model with custom module registration.
        # Input: model path or YAML config, task type -> Intermediate: registers modules and calls super() -> Output: URCDYOLO instance.
        register_urcd_modules()
        cfg_path = model or self.DEFAULT_CFG
        super().__init__(model=cfg_path, task=task, verbose=verbose)
