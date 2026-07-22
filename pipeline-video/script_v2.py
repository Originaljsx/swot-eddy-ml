"""Pipeline explainer v2 — built from REAL stage outputs (pass 049/091, held-out).

Every image is an actual intermediate array from the pipeline, not a schematic.
White background to match the figures; ~3 minutes; 7 scenes.
"""
from manim import *

INK = "#111111"
MUT = "#666666"
BLUE = "#0072B2"
VERM = "#D55E00"
VIOLET = "#7b3294"
GOLD = "#B8860B"
config.background_color = WHITE

A = "assets2/"


def img(path, height=6.0):
    im = ImageMobject(A + path)
    im.set(height=height)
    return im


def T(s, size=30, color=INK, bold=False):
    return Text(s, font_size=size, color=color, weight=BOLD if bold else NORMAL)


## Scene1_Input
class Scene1_Input(Scene):
    def construct(self):
        title = T("From a raw SWOT file to frontal eddies", 40, bold=True)
        sub = T("every frame below is real data from one held-out pass", 22, MUT)
        sub.next_to(title, DOWN)
        self.add_subcaption("From a raw SWOT file to frontal eddies — every frame is "
                            "real data from one held-out pass.", duration=3.5)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2))
        self.wait(1.2)
        self.play(FadeOut(title), FadeOut(sub))

        granule = img("step01_input_granule.png", 6.2).shift(LEFT * 3.3)
        txt = VGroup(
            T("One netCDF granule:", 28, bold=True),
            T("a pole-to-pole half-orbit,", 26),
            T("120 km wide, 2 km pixels.", 26),
            T("The Gulf Stream is a", 26),
            T("small band of it.", 26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).shift(RIGHT * 2.9)
        self.add_subcaption("The input: one netCDF granule — a pole-to-pole half-orbit "
                            "swath. Our study region is a small band of it.", duration=5)
        self.play(FadeIn(granule, shift=RIGHT * 0.4), FadeIn(txt))
        self.wait(2)
        self.play(FadeOut(granule), FadeOut(txt))


## Scene2_CropTile
class Scene2_CropTile(Scene):
    def construct(self):
        head = T("Branch 1 · py-eddy-tracker: prepare", 32, bold=True).to_edge(UP)
        self.play(Write(head))
        crop = img("step02_crop_band.png", 5.6).shift(LEFT * 4.2 + DOWN * 0.4)
        tilef = img("step03_pad_and_tiles.png", 5.6).shift(RIGHT * 0.2 + DOWN * 0.4)
        note = VGroup(
            T("crop to the region band", 24),
            T("pad 69 → 128 pixels", 24),
            T("128×128 tiles, 50% overlap", 24),
            T("(white = real data gaps)", 22, MUT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(RIGHT * 4.3 + DOWN * 0.4)
        self.add_subcaption("Crop the pass to the study band — note the white gaps: "
                            "nadir line, rain, editing.", duration=4)
        self.play(FadeIn(crop, shift=RIGHT * 0.3))
        self.wait(0.6)
        self.add_subcaption("Pad the 69 pixels to 128 and cut overlapping square tiles — "
                            "every eddy lands whole in at least one.", duration=4.5)
        self.play(FadeIn(tilef, shift=RIGHT * 0.3), FadeIn(note))
        self.wait(1.6)
        self.play(*map(FadeOut, (head, crop, tilef, note)))


## Scene3_Inpaint
class Scene3_Inpaint(Scene):
    def construct(self):
        head = T("Fill the gaps — PET needs closed contours", 32, bold=True).to_edge(UP)
        self.play(Write(head))
        pic = img("step04_inpaint.png", 4.6).shift(DOWN * 0.9)
        eq = MathTex(r"\nabla^4 u = 0 \ \ \text{in every gap}", color=INK).scale(0.95)
        eq.next_to(head, DOWN, buff=0.35)
        self.add_subcaption("Gaps are filled by solving the biharmonic equation — the "
                            "smoothest surface matching the data in value and slope at "
                            "every hole's edge.", duration=6)
        self.play(Write(eq))
        self.play(FadeIn(pic, shift=UP * 0.3))
        self.wait(2)
        warn = T("filled pixels are remembered — they are invented data", 22, VERM)
        warn.to_edge(DOWN)
        self.add_subcaption("Filled pixels are remembered: they are invented data, and "
                            "detections that rely on them get rejected.", duration=4.5)
        self.play(FadeIn(warn))
        self.wait(1.5)
        self.play(*map(FadeOut, (head, eq, pic, warn)))


## Scene4_PET
class Scene4_PET(Scene):
    def construct(self):
        head = T("PET finds every closed-contour eddy", 32, bold=True).to_edge(UP)
        self.play(Write(head))
        left = img("step05_pet_contours.png", 5.4).shift(LEFT * 3.4 + DOWN * 0.45)
        crit = VGroup(
            T("closed SSH contour around an extremum", 22),
            T("amplitude ≥ 1 cm", 22),
            T("speed radius ≥ 5 km", 22),
            T("shape error ≤ 85%", 22),
            T("reject: contour mostly on invented pixels", 22, MUT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).shift(RIGHT * 2.9 + UP * 0.9)
        self.add_subcaption("py-eddy-tracker scans SSH levels for closed contours passing "
                            "amplitude, size, and shape tests. Dashed gray contours sit on "
                            "invented pixels — auto-rejected.", duration=7)
        self.play(FadeIn(left, shift=RIGHT * 0.3), FadeIn(crit))
        self.wait(1.2)
        right = img("step06_pet_candidates.png", 5.4).move_to(left)
        res = T("15 candidate eddies — frontal-ness still unknown", 24, BLUE)
        res.shift(RIGHT * 2.9 + DOWN * 1.6)
        self.add_subcaption("After deduplication and the region filter: fifteen "
                            "closed-contour eddies. Which are frontal? PET cannot say.",
                            duration=5)
        self.play(FadeTransform(left, right), FadeIn(res))
        self.wait(1.8)
        self.play(*map(FadeOut, (head, right, crit, res)))


## Scene5_UNet
class Scene5_UNet(Scene):
    def construct(self):
        head = T("Branch 2 · U-Net: where is 'frontal'?", 32, bold=True).to_edge(UP)
        self.play(Write(head))
        chans = img("step07_unet_inputs.png", 3.4).shift(UP * 0.85)
        self.add_subcaption("A small U-Net reads four raw channels — height anomaly, "
                            "absolute topography, its gradient, and a validity mask. "
                            "No inpainting: gaps stay gaps.", duration=6.5)
        self.play(FadeIn(chans, shift=UP * 0.2))
        note = T("trained on 902 hand-confirmed frontal eddies", 22, MUT)
        note.next_to(chans, DOWN, buff=0.18)
        self.play(FadeIn(note))
        self.wait(1)
        self.wait(1)
        self.play(FadeOut(chans), FadeOut(note))

        front = img("step07b_adt_front.png", 5.2).shift(LEFT * 4.0 + DOWN * 0.5)
        ftxt = VGroup(
            T("'frontal' is defined by the jet:", 24),
            T("the ADT 0.35 m contour traces it;", 24),
            T("frontal eddies hug its inshore side", 24, GOLD),
            T("(this line bootstrapped the labels)", 20, MUT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).shift(RIGHT * 2.6 + DOWN * 0.5)
        self.add_subcaption("Frontal means: on the Gulf Stream. One ADT contour traces "
                            "the jet through every pass — frontal eddies hug its inshore "
                            "side. This line bootstrapped the training labels.",
                            duration=6.5)
        self.play(FadeIn(front, shift=RIGHT * 0.3), FadeIn(ftxt))
        self.wait(2)
        self.play(FadeOut(front), FadeOut(ftxt))

        prob = img("step08_unet_probability.png", 5.2).shift(LEFT * 4.0 + DOWN * 0.5)
        ptxt = VGroup(
            T("output: P(frontal) per pixel", 24),
            T("conformal λ̂ = 0.35:", 24),
            T("≤ 10% miss-rate guarantee", 24, VIOLET),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).shift(RIGHT * 2.6 + DOWN * 0.5)
        self.add_subcaption("The network's output: a frontal probability for every "
                            "pixel, with a conformally calibrated threshold guaranteeing "
                            "at most ten percent of frontal eddies are missed.",
                            duration=6)
        self.play(FadeIn(prob, shift=RIGHT * 0.3), FadeIn(ptxt))
        self.wait(2)
        self.play(*map(FadeOut, (head, prob, ptxt)))


## Scene6_Hybrid
class Scene6_Hybrid(Scene):
    def construct(self):
        head = T("The fusion: PET proposes, U-Net decides", 32, bold=True).to_edge(UP)
        self.play(Write(head))
        score = img("step09_hybrid_scoring.png", 5.2).shift(LEFT * 4.4 + DOWN * 0.5)
        hist = img("score_by_class.png", 3.4).shift(RIGHT * 2.4 + UP * 1.0)
        self.add_subcaption("Each PET contour is scored with the U-Net probability inside "
                            "it. The separation is almost perfectly bimodal: confirmed "
                            "frontals at one, everything else at zero.", duration=7)
        self.play(FadeIn(score, shift=RIGHT * 0.3))
        self.play(FadeIn(hist, shift=UP * 0.2))
        self.wait(1.2)
        final = img("step10_final_output.png", 5.2).move_to(score)
        metrics = VGroup(
            T("precision 0.80", 26, VERM),
            T("recall 0.96", 26, BLUE),
            T("F1 0.875", 26, INK, bold=True),
            T("(vs 0.58 for the U-Net alone)", 20, MUT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).shift(RIGHT * 2.4 + DOWN * 1.9)
        self.add_subcaption("The answer for this pass: five frontal eddies among fifteen. "
                            "On held-out cycles: precision point-eight, recall "
                            "point-nine-six, F1 point-eight-seven-five.", duration=7)
        self.play(FadeTransform(score, final), FadeIn(metrics))
        self.wait(2.2)
        self.play(*map(FadeOut, (head, final, hist, metrics)))


## Scene7_Outro
class Scene7_Outro(Scene):
    def construct(self):
        region = img("study_region_aoi5_labels_and_front.png", 5.6).shift(LEFT * 3.4)
        txt = VGroup(
            T("One command per pass:", 26, bold=True),
            Text("predict_hybrid.py GRANULE.nc", font_size=24, color=BLUE,
                 font="DejaVu Sans Mono"),
            T("→ every closed-contour eddy,", 24),
            T("   frontal ones flagged, with", 24),
            T("   coordinates and confidence", 24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.24).shift(RIGHT * 3.0)
        self.add_subcaption("Applied to fifty repeat cycles, the detections trace the "
                            "Gulf Stream from the Florida Straits to Cape Hatteras. One "
                            "command per pass: every eddy, frontal ones flagged.",
                            duration=8)
        self.play(FadeIn(region, shift=RIGHT * 0.3), FadeIn(txt))
        self.wait(3)
        fin = T("swot-eddy-ml", 26, MUT)
        self.play(FadeOut(region), FadeOut(txt))
        self.play(FadeIn(fin))
        self.wait(1.2)
        self.play(FadeOut(fin))
